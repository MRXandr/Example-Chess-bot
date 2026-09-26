"""Chess coaching chatbot: engine-verified moves, Groq-written explanations."""

import os
import re

import chess
import chess.svg
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import engine
from groq_client import GroqError, chat as groq_chat, list_models

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------- app state
class Settings:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.key_from_env = bool(self.api_key)


class Game:
    def __init__(self):
        self.board = chess.Board()
        self.orientation = chess.WHITE   # which side sits at the bottom
        self.transcript = []             # plain-text history for the LLM


settings = Settings()
game = Game()

# ------------------------------------------------------------ move parsing
SAN_PATTERN = re.compile(
    r"\b(?:O-O-O|0-0-0|O-O|0-0|"
    r"[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|"
    r"[a-h]x[a-h][1-8](?:=[QRBN])?|"
    r"[a-h][1-8](?:=[QRBN])?)[+#]?"
)
UCI_PATTERN = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbnQRBN]?\b")


def extract_move_tokens(text: str):
    """Pull move-looking tokens out of free text, in the order they appear."""
    cleaned = re.sub(r"\b\d+\s*\.+", " ", text)          # strip "1." "12..."
    tokens = []
    for match in list(UCI_PATTERN.finditer(cleaned)) + list(SAN_PATTERN.finditer(cleaned)):
        tokens.append((match.start(), match.group()))
    tokens.sort(key=lambda pair: pair[0])
    seen_positions = set()
    ordered = []
    for position, token in tokens:
        if any(position < end and position >= start for start, end in seen_positions):
            continue
        seen_positions.add((position, position + len(token)))
        ordered.append(token)
    return ordered


def apply_moves(board: chess.Board, tokens):
    """Try each token as SAN then UCI. Returns (applied_san, rejected)."""
    applied, rejected = [], []
    for token in tokens:
        move = None
        try:
            move = board.parse_san(token)
        except ValueError:
            try:
                candidate = chess.Move.from_uci(token.lower())
                if candidate in board.legal_moves:
                    move = candidate
            except ValueError:
                move = None
        if move is None:
            rejected.append(token)
            continue
        applied.append(board.san(move))
        board.push(move)
    return applied, rejected


# --------------------------------------------------------------- rendering
def board_svg() -> str:
    last = game.board.move_stack[-1] if game.board.move_stack else None
    check_square = None
    if game.board.is_check():
        check_square = game.board.king(game.board.turn)
    return chess.svg.board(
        board=game.board,
        size=440,
        orientation=game.orientation,
        lastmove=last,
        check=check_square,
        coordinates=True,
    )


def move_history_san():
    replay = chess.Board()
    history = []
    for move in game.board.move_stack:
        history.append(replay.san(move))
        replay.push(move)
    return history


def pgn_line() -> str:
    history = move_history_san()
    parts = []
    for index, san in enumerate(history):
        if index % 2 == 0:
            parts.append(f"{index // 2 + 1}.{san}")
        else:
            parts.append(san)
    return " ".join(parts) if parts else "(no moves yet)"


def material_summary() -> str:
    counts = []
    for colour, label in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        pieces = []
        for piece_type, name in (
            (chess.QUEEN, "Q"), (chess.ROOK, "R"),
            (chess.BISHOP, "B"), (chess.KNIGHT, "N"), (chess.PAWN, "P"),
        ):
            count = len(game.board.pieces(piece_type, colour))
            if count:
                pieces.append(f"{count}{name}")
        counts.append(f"{label}: {' '.join(pieces) or 'bare king'}")
    return " | ".join(counts)


def game_over_text():
    board = game.board
    if not board.is_game_over():
        return None
    outcome = board.outcome()
    if outcome and outcome.winner is not None:
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        return f"Game over — checkmate, {winner} wins."
    reason = outcome.termination.name.replace("_", " ").lower() if outcome else "draw"
    return f"Game over — draw ({reason})."


def state_payload(message: str = "", role: str = "bot", error: str = ""):
    return {
        "svg": board_svg(),
        "fen": game.board.fen(),
        "turn": "White" if game.board.turn == chess.WHITE else "Black",
        "history": move_history_san(),
        "pgn": pgn_line(),
        "in_check": game.board.is_check(),
        "game_over": game.board.is_game_over(),
        "message": message,
        "role": role,
        "error": error,
        "has_key": bool(settings.api_key),
        "model": settings.model,
        "key_from_env": settings.key_from_env,
        "orientation": "white" if game.orientation == chess.WHITE else "black",
    }


# -------------------------------------------------------------- suggestion
def build_analysis():
    """Run the engine and return (candidates, text_block)."""
    candidates = engine.search(game.board, max_depth=3, top_n=4, time_limit=5.0)
    if not candidates:
        return [], "No legal moves available."

    lines = []
    for index, (move, score) in enumerate(candidates):
        san = game.board.san(move)
        entry = f"{index + 1}. {san} (eval {engine.score_to_text(score)})"
        if index == 0:
            reply = engine.best_reply(game.board, move, depth=2)
            if reply:
                pushed = game.board.copy()
                pushed.push(move)
                entry += f" — likely reply: {pushed.san(reply)}"
        lines.append(entry)
    return candidates, "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a friendly, sharp chess coach speaking to a club-level player.\n"
    "You are given the exact position, the move history, and engine analysis "
    "listing the strongest legal candidate moves with evaluations in pawns "
    "(positive means good for the side to move).\n\n"
    "Rules you must follow:\n"
    "- Recommend ONLY a move from the candidate list. Never invent a move.\n"
    "- Open with a bold line exactly like: **Play: Nf3**\n"
    "- Then 2-4 short sentences on why: the idea, the threat, the plan.\n"
    "- Then a line starting with 'Watch out for:' naming the opponent's best "
    "reply or the main danger in the position.\n"
    "- Optionally end with one short alternative from the list.\n"
    "- Plain language, no engine jargon dumps, under 180 words.\n"
)


def coach_reply(extra_question: str = ""):
    over = game_over_text()
    if over:
        return over

    candidates, analysis_text = build_analysis()
    if not candidates:
        return "There are no legal moves in this position."

    side = "White" if game.board.turn == chess.WHITE else "Black"
    top_move_san = game.board.san(candidates[0][0])

    if not settings.api_key:
        return (
            f"**Play: {top_move_san}**\n\n"
            f"(Engine-only suggestion — add a Groq API key in Settings for a full explanation.)\n\n"
            f"Candidates:\n{analysis_text}"
        )

    user_content = (
        f"Side to move: {side}\n"
        f"FEN: {game.board.fen()}\n"
        f"Moves so far: {pgn_line()}\n"
        f"Material: {material_summary()}\n"
        f"In check: {'yes' if game.board.is_check() else 'no'}\n\n"
        f"Engine candidates (best first):\n{analysis_text}\n"
    )
    if extra_question:
        user_content += f"\nThe player also asks: {extra_question}\n"

    try:
        return groq_chat(
            settings.api_key,
            settings.model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except GroqError as exc:
        return (
            f"**Play: {top_move_san}**\n\n"
            f"(AI explanation unavailable: {exc})\n\n"
            f"Candidates:\n{analysis_text}"
        )


def answer_question(question: str) -> str:
    """Free-form chess question about the current position — no move applied."""
    _, analysis_text = build_analysis()
    if not settings.api_key:
        return (
            "Add a Groq API key in Settings and I can answer questions about the "
            f"position. Engine view right now:\n{analysis_text}"
        )
    user_content = (
        f"Position FEN: {game.board.fen()}\n"
        f"Moves so far: {pgn_line()}\n"
        f"Engine candidates:\n{analysis_text}\n\n"
        f"Player's question: {question}"
    )
    try:
        return groq_chat(
            settings.api_key,
            settings.model,
            [
                {"role": "system", "content":
                    "You are a helpful chess coach. Answer the player's question about "
                    "this position clearly and briefly. If you name moves, use only legal "
                    "ones consistent with the FEN and engine list. Under 180 words."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
        )
    except GroqError as exc:
        return f"Couldn't reach Groq: {exc}"


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state_payload())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    game.board.reset()
    game.transcript.clear()
    return jsonify(state_payload("New game. You're playing from the starting position — tell me the moves as they happen."))


@app.route("/api/undo", methods=["POST"])
def api_undo():
    if not game.board.move_stack:
        return jsonify(state_payload("Nothing to take back yet."))
    undone = game.board.pop()
    return jsonify(state_payload(f"Took back {undone.uci()}. Position restored."))


@app.route("/api/flip", methods=["POST"])
def api_flip():
    game.orientation = not game.orientation
    return jsonify(state_payload())


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(silent=True) or {}
    key = (data.get("api_key") or "").strip()
    model = (data.get("model") or "").strip()
    if key:
        settings.api_key = key
        settings.key_from_env = False
    if model:
        settings.model = model
    return jsonify(state_payload("Settings saved."))


@app.route("/api/models", methods=["POST"])
def api_models():
    data = request.get_json(silent=True) or {}
    key = (data.get("api_key") or "").strip() or settings.api_key
    try:
        return jsonify({"models": list_models(key)})
    except GroqError as exc:
        return jsonify({"models": [], "error": str(exc)}), 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify(state_payload("Say something — a move, or ask me a question."))

    lowered = text.lower().strip(" .!?")

    if lowered in {"reset", "new game", "restart", "start over"}:
        game.board.reset()
        return jsonify(state_payload("Fresh board. Go ahead."))
    if lowered in {"undo", "take back", "takeback"}:
        if game.board.move_stack:
            game.board.pop()
            return jsonify(state_payload("Took that back."))
        return jsonify(state_payload("Nothing to undo."))
    if lowered in {"fen"}:
        return jsonify(state_payload(f"FEN: `{game.board.fen()}`"))
    if lowered in {"pgn", "moves", "history"}:
        return jsonify(state_payload(f"Moves so far: {pgn_line()}"))
    if "i am black" in lowered or "i'm black" in lowered or "play as black" in lowered:
        game.orientation = chess.BLACK
        return jsonify(state_payload("Board flipped — you're Black."))
    if "i am white" in lowered or "i'm white" in lowered or "play as white" in lowered:
        game.orientation = chess.WHITE
        return jsonify(state_payload("Board set — you're White."))
    if lowered in {"hint", "suggest", "help", "what should i play", "suggestion", "?"}:
        return jsonify(state_payload(coach_reply()))

    tokens = extract_move_tokens(text)
    applied, rejected = apply_moves(game.board, tokens)

    if not applied:
        # No moves found: treat it as a question about the position.
        if len(text.split()) >= 3 or text.endswith("?"):
            return jsonify(state_payload(answer_question(text)))
        note = f"I couldn't read that as a legal move: {', '.join(rejected) or text}"
        return jsonify(state_payload(
            f"{note}\n\nTry standard notation like `e4`, `Nf3`, `Bxc6+`, `O-O`, or `e2e4`.",
            error="illegal",
        ))

    prefix = "Played: " + ", ".join(applied) + "."
    if rejected:
        prefix += f" (Ignored: {', '.join(rejected)} — not legal here.)"

    over = game_over_text()
    if over:
        return jsonify(state_payload(f"{prefix}\n\n{over}"))

    return jsonify(state_payload(f"{prefix}\n\n{coach_reply()}"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)