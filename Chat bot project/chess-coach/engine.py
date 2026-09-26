"""A small but honest chess engine: alpha-beta negamax + quiescence search.

Its job is to guarantee that every move we suggest is legal and not an
obvious blunder. The LLM only chooses among the candidates produced here.
"""

import time
import chess

MATE_SCORE = 100_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables, written from White's point of view, rank 8 first.
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_MID_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

KING_END_PST = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}


def _is_endgame(board: chess.Board) -> bool:
    """Rough endgame test: little heavy material left on the board."""
    heavy = 0
    for piece_type, weight in (
        (chess.QUEEN, 9), (chess.ROOK, 5),
        (chess.BISHOP, 3), (chess.KNIGHT, 3),
    ):
        heavy += weight * len(board.pieces(piece_type, chess.WHITE))
        heavy += weight * len(board.pieces(piece_type, chess.BLACK))
    return heavy <= 16


def evaluate(board: chess.Board) -> int:
    """Score in centipawns, from the point of view of the side to move."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if board.can_claim_fifty_moves() or board.is_repetition(3):
        return 0

    endgame = _is_endgame(board)
    white_score = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        if piece.piece_type == chess.KING:
            table = KING_END_PST if endgame else KING_MID_PST
        else:
            table = PST[piece.piece_type]
        # square_mirror() turns a square into the index of the top-down table.
        index = chess.square_mirror(square) if piece.color == chess.WHITE else square
        value += table[index]
        white_score += value if piece.color == chess.WHITE else -value

    # Small bonus for having options.
    mobility = board.legal_moves.count()
    white_score += 2 * mobility if board.turn == chess.WHITE else -2 * mobility

    return white_score if board.turn == chess.WHITE else -white_score


def _move_order_key(board: chess.Board, move: chess.Move) -> int:
    """Search captures and checks first so alpha-beta prunes more."""
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_value = PIECE_VALUES[victim.piece_type] if victim else 100  # en passant
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
        score += 10_000 + victim_value * 10 - attacker_value
    if move.promotion:
        score += 9_000 + PIECE_VALUES.get(move.promotion, 0)
    if board.gives_check(move):
        score += 500
    return score


def _quiesce(board: chess.Board, alpha: int, beta: int, depth: int = 4) -> int:
    """Keep searching captures so we don't stop mid-trade and misjudge."""
    stand_pat = evaluate(board)
    if depth == 0 or stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)

    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    captures.sort(key=lambda m: _move_order_key(board, m), reverse=True)

    for move in captures:
        board.push(move)
        score = -_quiesce(board, -beta, -alpha, depth - 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int, deadline: float) -> int:
    if time.time() > deadline:
        raise TimeoutError

    if board.is_checkmate():
        return -MATE_SCORE + ply           # prefer faster mates
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        return _quiesce(board, alpha, beta)

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: _move_order_key(board, m), reverse=True)

    best = -MATE_SCORE * 2
    for move in moves:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def search(board: chess.Board, max_depth: int = 3, top_n: int = 4, time_limit: float = 5.0):
    """Iterative deepening. Returns [(move, score_in_centipawns), ...] best first.

    Scores are from the point of view of the side to move.
    """
    deadline = time.time() + time_limit
    root_moves = list(board.legal_moves)
    if not root_moves:
        return []

    results = [(m, 0) for m in root_moves]

    for depth in range(1, max_depth + 1):
        scored = []
        try:
            ordered = [m for m, _ in results]
            alpha = -MATE_SCORE * 2
            for move in ordered:
                board.push(move)
                score = -_negamax(board, depth - 1, -MATE_SCORE * 2, MATE_SCORE * 2, 1, deadline)
                board.pop()
                scored.append((move, score))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            results = scored
        except TimeoutError:
            break

    return results[:top_n]


def score_to_text(score: int) -> str:
    """Human-readable evaluation for the side to move."""
    if abs(score) > MATE_SCORE - 1000:
        moves_to_mate = (MATE_SCORE - abs(score) + 1) // 2
        return f"mate in {max(moves_to_mate, 1)}" if score > 0 else f"mated in {max(moves_to_mate, 1)}"
    pawns = score / 100.0
    return f"{pawns:+.2f}"


def best_reply(board: chess.Board, move: chess.Move, depth: int = 2):
    """The opponent's likely answer to a candidate move, for context."""
    board.push(move)
    try:
        replies = search(board, max_depth=depth, top_n=1, time_limit=2.0)
        return replies[0][0] if replies else None
    finally:
        board.pop()