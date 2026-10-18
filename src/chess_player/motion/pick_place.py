"""Turn a chess move into piece transfers, and execute a transfer as hover/descend/grasp/lift/travel/place.

All IK for a transfer is solved before the arm moves, so an unreachable target fails without moving anything.
Vertical and horizontal segments are straight Cartesian lines (dense IK), so the jaws go straight down and
travel level.

Ported from Nithya-shree182/chess-player (sim/planner.py, sim/chess_move.py), adapted to this calibration:
- travel height per transfer = tallest piece near the path + how far the carried piece hangs below the tip +
  TRAVEL_CLEAR_MM (instead of always Z_SAFE): less stretch and tilt at far squares;
- collision check of the swept gripper (capsule above the tip) and the carried piece against every other
  piece on the board and in the graveyard; a hit raises Blocked before anything moves;
- PickPlace.check() returns a per-step feasibility report.
"""
from dataclasses import dataclass

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import mm_to_square, square_center_mm
from chess_player.motion.arm import look_pose
from chess_player.motion.ik import Unreachable, solve_or_raise


@dataclass
class Transfer:
    piece: str  # python-chess symbol, e.g. "P", "n"
    src: tuple  # board mm
    dst: tuple
    label: str  # for logs / commentary


class Blocked(Unreachable):
    """The swept gripper or the carried piece would hit another piece."""


class PieceDropped(RuntimeError):
    """The piece slipped out of the jaws while it was being carried."""


class GraspFailed(RuntimeError):
    pass


class Slots:
    """Off-board places: one graveyard slot per captured piece, and the spare promotion pieces."""

    def __init__(self, board_to_robot=None):
        slots = list(C.GRAVEYARD)
        if board_to_robot is not None:  # nearest the robot base first: shortest, most reliable moves
            slots.sort(key=lambda xy: float(np.linalg.norm(board_to_robot(*xy)[:2])))
        self.free_graveyard = slots
        self.spares = {k: list(v) for k, v in C.SPARES.items()}
        self.occupied = {}  # graveyard xy -> piece symbol (obstacles for later moves)

    def graveyard(self, symbol="p"):
        if not self.free_graveyard:
            raise RuntimeError("graveyard full")
        xy = self.free_graveyard.pop(0)
        self.occupied[xy] = symbol
        return xy

    def spare(self, symbol):
        if not self.spares.get(symbol):
            raise RuntimeError(f"no spare {symbol} left for promotion: place one by hand")
        return self.spares[symbol].pop(0)


def _xy(sq):
    return tuple(square_center_mm(sq))


def surface_z(xy):
    """Height (board mm) of the surface a piece at xy stands on: the board, or the table beyond its outer edge."""
    x, y = xy
    lo, hi = -C.EDGE_MM, C.PLAY_MM + C.EDGE_MM
    return 0.0 if (lo <= x <= hi and lo <= y <= hi) else -C.BOARD_THICKNESS_MM


def plan_move(board: chess.Board, move: chess.Move, slots: Slots) -> list[Transfer]:
    """Transfers that turn the physical position `board` into board.push(move). `board` is not modified."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError(f"no piece on {chess.square_name(move.from_square)}")
    out = []
    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rank = chess.square_rank(move.from_square)
        r_from, r_to = chess.square(7 if kingside else 0, rank), chess.square(5 if kingside else 3, rank)
        rook = board.piece_at(r_from).symbol()
        # Order so each piece is set down with the square on its h-file side (where the moving finger opens) empty:
        # kingside = rook to f1 first (g1 still empty), then king to g1 (h1 now empty); queenside = king to c1
        # first (d1 empty), then rook to d1 (e1 now empty).
        rook_t = Transfer(rook, _xy(r_from), _xy(r_to), f"rook {chess.square_name(r_from)}{chess.square_name(r_to)}")
        king_t = Transfer(piece.symbol(), _xy(move.from_square), _xy(move.to_square), f"king {move.uci()}")
        out += [rook_t, king_t] if kingside else [king_t, rook_t]
        return out

    if board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    elif board.is_capture(move):
        cap_sq = move.to_square
    else:
        cap_sq = None
    if cap_sq is not None:
        victim = board.piece_at(cap_sq).symbol()
        out.append(Transfer(victim, _xy(cap_sq), slots.graveyard(victim), f"capture {victim} on {chess.square_name(cap_sq)}"))

    if move.promotion:
        new = chess.Piece(move.promotion, piece.color).symbol()
        out.append(Transfer(piece.symbol(), _xy(move.from_square), slots.graveyard(piece.symbol()), "promoting pawn off the board"))
        out.append(Transfer(new, slots.spare(new), _xy(move.to_square), f"promote to {new} on {chess.square_name(move.to_square)}"))
    else:
        out.append(Transfer(piece.symbol(), _xy(move.from_square), _xy(move.to_square), f"{piece.symbol()} {move.uci()}"))
    return out


def obstacles_for(board: chess.Board, slots: Slots, transfers):
    """Yield (transfer, obstacles) with obstacles = {xy: symbol} of every other piece at that moment."""
    occ = {_xy(sq): p.symbol() for sq, p in board.piece_map().items()}
    occ.update(slots.occupied)
    for t in transfers:
        occ.pop(t.dst, None)  # a graveyard slot was pre-booked for this piece; it's empty until we put it there
        obst = {xy: s for xy, s in occ.items() if xy != t.src}
        yield t, obst
        occ.pop(t.src, None)
        occ[t.dst] = t.piece


def _seg_dist(c, a, b):
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    ab = b - a
    t = np.clip(np.dot(c - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return float(np.linalg.norm(c - (a + t * ab)))


def _need_height(xy, obstacles, held_r):
    """Lowest tip height (board mm) at xy so the carried piece (held at its base) and the fingers next to it clear
    every piece whose footprint overlaps."""
    need = 0.0
    for (ox, oy), sym in obstacles.items():
        o = C.PIECES[sym.lower()]
        if np.hypot(xy[0] - ox, xy[1] - oy) < held_r + C.FINGER_LEFT_MM + o["diameter"] / 2 + C.CORRIDOR_MARGIN_MM:
            need = max(need, o["height"] + surface_z((ox, oy)) + C.GRASP_HEIGHT_MM + C.TRAVEL_CLEAR_MM)
    return need


def _polyline(points, step):
    pts = [np.asarray(points[0], float)]
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(pts)


def plan_travel(src, dst, z0, z1, obstacles, held_r, step=10.0):
    """Carry path (board-mm xyz points of the piece centre) from (src, z0) to (dst, z1), shaped like the
    demonstrations: straight up to a high travel level, across with a gentle hump, straight down over the target.

    Lift H = ARC_PER_MM x distance, clamped to ARC_MIN_MM..ARC_MAX_MM; ARC_RISE_FRACTION of it is the vertical part.
    The travel is raised wherever a piece is in the way, and kept rise-then-fall. Routes tried: straight, or bent
    half a square / one square to either side; the cheapest (peak height + extra length) wins.
    Returns (points, info) or (None, reason).
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    d = dst - src
    L = float(np.linalg.norm(d))
    perp = np.array([-d[1], d[0]]) / (L + 1e-9)
    mid = (src + dst) / 2
    routes = [("straight", [src, dst])]
    if obstacles and L > C.SQUARE_MM * 0.9:
        for k in (0.5, -0.5, 1.0, -1.0):
            routes.append((f"bend {k:+.1f} sq", [src, mid + perp * k * C.SQUARE_MM, dst]))
    best = None
    for name, corners in routes:
        xy = _polyline(corners, step)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        length = float(seg[-1])
        s = seg / (length + 1e-9)
        H = float(np.clip(C.ARC_PER_MM * length, C.ARC_MIN_MM, C.ARC_MAX_MM))
        level = max(z0, z1) + C.ARC_RISE_FRACTION * H
        z = level + (1 - C.ARC_RISE_FRACTION) * H * np.sin(np.pi * s)
        need = np.array([_need_height(p, obstacles, held_r) for p in xy]) if obstacles else np.zeros(len(xy))
        z = np.maximum(z, need)
        z = np.minimum(np.maximum.accumulate(z), np.maximum.accumulate(z[::-1])[::-1])  # rise, then fall
        if z.max() > C.Z_SAFE:
            continue
        rise = np.c_[np.tile(xy[0], (8, 1)), np.linspace(z0, z[0], 8)]  # straight up
        fall = np.c_[np.tile(xy[-1], (8, 1)), np.linspace(z[-1], z1, 8)]  # straight down
        pts = np.r_[rise[:-1], np.c_[xy, z], fall[1:]]
        total = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        cost = float(z.max()) + C.DETOUR_COST * (length - L)
        if best is None or cost < best[0]:
            best = (cost, name, pts, total, float(z.max()))
    if best is None:
        return None, "pieces in the way are taller than the safe travel height"
    _, name, pts, total, peak = best
    return pts, dict(route=name, peak_mm=round(peak, 1), length_mm=round(total, 1))


def collisions(path, obstacles, held_below, held_r, piece_d, open_pct=None):
    """path: [(board-mm xyz of the tip, u, holding)], u = unit board-xy vector toward the moving finger.
    Fingers: a strip from FINGER_LEFT_MM left of the tip to (piece width + FINGER_WIDTH_MM) right of it, from the
    tip up FINGER_LENGTH_MM; above that the gripper body (BODY_RADIUS_MM). The held piece is centred HELD_OFFSET_MM
    right of the tip, its base held_below under the tip. Returns {(symbol, xy)} hit."""
    hits = set()
    wide_right = piece_d + C.FINGER_WIDTH_MM  # jaws open around a piece centred on its square (approach)
    if open_pct is not None:  # opened less than GRIPPER_OPEN (tight-spot grasp): the moving finger stays closer
        wide_right -= max(0.0, C.GRIPPER_OPEN - open_pct) * C.JAW_MM_PER_PCT
    # after the grasp the piece sits HELD_OFFSET_MM from the tip and the moving finger hugs it (or opens a little:
    # rest[1] = how far it has moved off the piece there, default RELEASE_GAP_MM)
    for (x, y, z), u, holding, *rest in path:
        gap = rest[1] if len(rest) > 1 else C.RELEASE_GAP_MM
        right = C.HELD_OFFSET_MM + held_r + gap + C.FINGER_WIDTH_MM if (rest and rest[0]) else wide_right
        p = np.array([x, y])
        w = np.array([-u[1], u[0]])  # across the jaw axis
        for (ox, oy), sym in obstacles.items():
            o = C.PIECES[sym.lower()]
            r = o["diameter"] / 2
            top = o["height"] + surface_z((ox, oy))
            c = np.array([ox, oy])
            s_, w_ = float(np.dot(c - p, u)), float(np.dot(c - p, w))  # obstacle centre in the finger frame
            in_fingers = -C.FINGER_LEFT_MM - r < s_ < right + r and abs(w_) < C.FINGER_HALF_DEPTH_MM + r
            if in_fingers and z < top:
                hits.add((sym, (round(ox), round(oy))))
            elif np.linalg.norm(c - p) < 0.3 * o["diameter"] + C.BODY_RADIUS_MM and z + C.FINGER_LENGTH_MM < top:
                # body vs the piece's (narrow) top part
                hits.add((sym, (round(ox), round(oy))))
            elif holding and np.linalg.norm(c - (p + u * C.HELD_OFFSET_MM)) < held_r + r and z - held_below < top:
                hits.add((sym, (round(ox), round(oy))))
    return hits


class PickPlace:
    def __init__(self, arm, ik, board_to_robot, verify_grasp=True, log=print):
        self.arm, self.ik, self.T = arm, ik, board_to_robot
        self.verify_grasp = verify_grasp
        self.log = log

    def _target(self, p):
        """Board-mm point -> robot-frame IK target, with the measured whole-motion trim (ARM_TARGET_OFFSET_MM)."""
        dx, dy = C.ARM_TARGET_OFFSET_MM
        return self.T(p[0] + dx, p[1] + dy, p[2])

    # ---------------------------------------------------------------- planning
    def _line(self, xy0, z0, xy1, z1, seed, max_tilt, step_mm=8.0, pts=None):
        """Joint waypoints (arm joints only) along a straight line in board coordinates. Appends the board-mm
        points to `pts` if given (for the collision check)."""
        p0, p1 = np.array([*xy0, z0]), np.array([*xy1, z1])
        n = max(2, int(np.linalg.norm(p1 - p0) / step_mm) + 1)
        qs, q = [], seed
        for s in np.linspace(0, 1, n):
            p = p0 + (p1 - p0) * s
            q = solve_or_raise(self.ik, self._target(p), q, max_tilt, what=f"point {np.round(p)} mm")
            qs.append(q)
            if pts is not None:
                pts.append(p)
        return qs

    @staticmethod
    def _points(xy0, z0, xy1, z1, step_mm=8.0):
        """The board-mm points _line() would visit."""
        p0, p1 = np.array([*xy0, z0]), np.array([*xy1, z1])
        n = max(2, int(np.linalg.norm(p1 - p0) / step_mm) + 1)
        return [p0 + (p1 - p0) * s for s in np.linspace(0, 1, n)]

    def _path(self, points, seed, max_tilt=90.0):
        """IK along explicit board-mm xyz points."""
        qs, q = [], seed
        for p in points:
            q = solve_or_raise(self.ik, self._target(p), q, max_tilt, what=f"point {np.round(p)} mm")
            qs.append(q)
        return qs

    def plan_transfer(self, t: Transfer, obstacles=None):
        """Plan with the demonstrated jaw shift; in a crowded spot (the fixed finger would touch a neighbour) retry
        with a smaller shift before giving up."""
        last = None
        tries = [(s, False) for s in (C.JAW_SHIFT_MM, C.JAW_SHIFT_MM - 2.5, C.JAW_SHIFT_MM - 5.0)]
        tries += [(s, True) for s in C.TIGHT_JAW_SHIFT_MM]  # the crowded-demo grasp (only if the normal one is blocked)
        for shift, tight in tries:
            try:
                plan = self._plan_transfer(t, obstacles, shift, tight)
                plan["info"]["jaw_shift_mm"] = shift
                plan["info"]["grasp"] = "tight" if tight else "normal"
                return plan
            except Blocked as e:
                last = e
        raise last

    def _plan_transfer(self, t: Transfer, obstacles, jaw_shift, tight=False):
        """IK for every waypoint + collision check. Raises Unreachable / Blocked; returns the plan dict.

        Sequence (board mm; the piece stays centred on its squares, the tip is offset toward the fixed finger):
          hover high over the square -> down to just above the piece -> step sideways JAW_SHIFT_MM toward the fixed
          finger (visible "go left") -> down to the piece's base -> grasp -> lift LIFT_MM -> arc / detour to the
          destination -> down to the base -> release -> straight up to Z_SAFE.
        tight (crowded spot, from the crowded demos): open TIGHT_GRIPPER_OPEN, the sideways step above the tallest
        neighbour, release with only TIGHT_RELEASE_OPEN_PCT. Where the piece is picked / put down is the same.
        """
        obstacles = obstacles or {}
        spec = C.PIECES[t.piece.lower()]
        held_below, held_r = C.GRASP_HEIGHT_MM, spec["diameter"] / 2
        z_grasp = C.GRASP_HEIGHT_MM + surface_z(t.src)
        z_place = C.GRASP_HEIGHT_MM + C.Z_PLACE_LIFT + surface_z(t.dst)
        z_side = spec["height"] + surface_z(t.src) + C.SIDE_STEP_CLEAR_MM  # just above the piece's top
        open_pct = C.TIGHT_GRIPPER_OPEN if tight else C.GRIPPER_OPEN
        release_pct = C.TIGHT_RELEASE_OPEN_PCT if tight else C.RELEASE_OPEN_PCT
        release_gap = release_pct * C.JAW_MM_PER_PCT if tight else C.RELEASE_GAP_MM
        if tight:  # step sideways above everything near the square, then straight down
            near = [C.PIECES[s.lower()]["height"] + surface_z(xy) for xy, s in (obstacles or {}).items()
                    if np.hypot(xy[0] - t.src[0], xy[1] - t.src[1]) < 1.6 * C.SQUARE_MM]
            z_side = min(C.Z_SAFE - 5.0, max([z_side - C.SIDE_STEP_CLEAR_MM] + near) + C.SIDE_STEP_CLEAR_MM)
        seed = np.array([look_pose()[j] for j in C.ARM_JOINTS])
        seed[4] = C.WRIST_ROLL_DEG
        (src_xy, u_src), (_, u_dst) = self._jaw_shifted(t.src, z_grasp, seed, jaw_shift), self._jaw_shifted(t.dst, z_place, seed)
        # once gripped the piece sits HELD_OFFSET_MM from the tip (closing pushed it against the fixed finger),
        # so placing needs a smaller offset than grasping
        held_src = tuple(np.asarray(src_xy) + C.HELD_OFFSET_MM * u_src)  # where the piece is after the grasp
        dst_xy = tuple(np.asarray(t.dst, float) - C.HELD_OFFSET_MM * u_dst)
        p_down, p_side, p_dg = [], [], []
        down = self._line(t.src, C.Z_SAFE, t.src, z_side, seed, 90.0, pts=p_down)  # centred, jaws open
        # The arm is COMMANDED with the measured pick correction (PICK_EXTRA_*: toward the robot's base, toward the
        # fixed finger, lower; eased in over the side step / descent and out over the lift); the points kept for the
        # collision check stay where the fingers really end up (the correction only cancels the arm's error there).
        base_xy = -(self.T.R.T @ self.T.t)[:2] * 1000.0  # robot base in board mm
        to_arm = base_xy - np.asarray(t.src, float)
        to_arm /= np.linalg.norm(to_arm)
        corr = np.r_[-C.PICK_EXTRA_LEFT_MM * u_src + C.PICK_EXTRA_TOWARD_ARM_MM * to_arm, -C.PICK_EXTRA_DOWN_MM]
        ramp = lambda pts, a, b, z=True: [p + corr * np.r_[f, f, f if z else 0] for p, f in zip(pts, np.linspace(a, b, len(pts)))]
        p_side = self._points(t.src, z_side, src_xy, z_side, step_mm=3.0)  # "go left"
        side = self._path(ramp(p_side, 0, 1, z=False), down[-1])
        p_dg = self._points(src_xy, z_side, src_xy, z_grasp)  # down to the base
        dg = self._path([p + np.r_[corr[:2], corr[2] * f] for p, f in zip(p_dg, np.linspace(0, 1, len(p_dg)))], side[-1])
        for q in dg[-3:]:  # the last stretch of the descent must be near vertical
            self._tilt_check(q, t.label)
        z_lift0, z_lift1 = z_grasp + C.LIFT_MM, z_place + C.LIFT_MM
        p_down2, p_up2 = [], []  # (filled below)
        p_up = self._points(src_xy, z_grasp, src_xy, z_lift0, step_mm=4.0)
        up = self._path(ramp(p_up, 1, 0), dg[-1])  # lift; the correction eases out
        # the carried piece's path (its centre is JAW_SHIFT_MM right of the tip) decides height and route
        piece_path, tinfo = plan_travel(held_src, t.dst, z_lift0, z_lift1, obstacles, held_r)
        if piece_path is None:
            raise Blocked(f"{t.label}: {tinfo}")
        n = max(1, len(piece_path) - 1)
        u_tr = [(u_src * (1 - i / n) + u_dst * (i / n)) / np.linalg.norm(u_src * (1 - i / n) + u_dst * (i / n))
                for i in range(len(piece_path))]
        p_travel = [np.r_[p[:2] - C.HELD_OFFSET_MM * u, p[2]] for p, u in zip(piece_path, u_tr)]
        # place-only trim (PLACE_EXTRA_OFFSET_MM): commanded positions only, eased in over the carry; the points kept
        # for the collision check stay where the piece really goes
        place_off = np.r_[np.asarray(C.PLACE_EXTRA_OFFSET_MM, float), 0.0]
        travel = self._path([p + place_off * f for p, f in zip(p_travel, np.linspace(0, 1, len(p_travel)))], up[-1])
        p_down2 = self._points(dst_xy, z_lift1, dst_xy, z_place, step_mm=4.0)
        down2 = self._path([p + place_off for p in p_down2], travel[-1])
        for q in down2[-3:]:
            self._tilt_check(q, t.label)
        p_up2 = self._points(dst_xy, z_place, dst_xy, C.Z_SAFE)
        up2 = self._path([p + place_off for p in p_up2], down2[-1])  # retreat, jaws open
        if obstacles:
            held_gap = 0.0 if tight else C.RELEASE_GAP_MM  # while holding, the moving finger hugs the piece
            path = ([(p, u_src, False, False) for p in p_down + p_side + p_dg]
                    + [(p, u_src, True, True, held_gap) for p in p_up]
                    + [(p, u, True, True, held_gap) for p, u in zip(p_travel, u_tr)]
                    + [(p, u_dst, True, True, held_gap) for p in p_down2]
                    + [(p, u_dst, False, True, release_gap) for p in p_up2])
            hits = collisions(path, obstacles, held_below, held_r, spec["diameter"], open_pct if tight else None)
            if hits:
                def where(xy):
                    sq = mm_to_square(*xy)
                    return chess.square_name(sq) if sq is not None else f"{xy} mm"

                names = [f"{chess.piece_name(chess.Piece.from_symbol(sym).piece_type)} on {where(xy)}"
                         for sym, xy in sorted(hits)]
                raise Blocked(f"{t.label}: the fingers would touch the " + " and the ".join(names))
        tilts = [float(np.degrees(np.arccos(np.clip(-self.ik.fk(q)[2, 2], -1, 1)))) for q in (dg[-1], down2[-1])]
        n_pts = sum(map(len, (p_down, p_side, p_dg, p_up, p_travel, p_down2, p_up2)))
        travel_s = max(C.MIN_MOVE_S, tinfo["length_mm"] / C.TRAVEL_SPEED_MM_S)
        info = dict(step=t.label, piece=t.piece, route=tinfo["route"], z_travel_mm=tinfo["peak_mm"],
                    max_grasp_tilt_deg=round(max(tilts), 1),
                    est_seconds=round(float(5 * C.DESCEND_S + travel_s + 2 * C.GRIP_S + 3 * C.MIN_MOVE_S), 1),
                    waypoints=n_pts)
        approach = down + side + dg
        return dict(approach=approach, abort_up=approach[::-1], up=up, travel=travel, travel_s=travel_s,
                    open=open_pct, release_pct=release_pct,
                    down2=down2, up2=up2, grip=spec["grip"], info=info)

    def _jaw_shifted(self, xy, z, seed, shift=None):
        """Tip xy (board mm) that puts the piece centred between the fixed (left) finger and the moving one:
        the piece centre minus JAW_SHIFT_MM along the direction the moving finger opens at that square.
        Also returns that direction (unit, board xy). JAW_SHIFT_MM < 0 shifts the other way."""
        q = solve_or_raise(self.ik, self.T(*xy, z), seed, 90.0, what=f"jaw direction at {np.round(xy)}")

        def jaw_and_tip(g):
            self.ik._set(q)
            self.ik.robot.set_joint("gripper", g)
            self.ik.robot.update_kinematics()
            return self.ik.robot.get_T_world_frame("moving_jaw_so101_v1_link"), self.ik.robot.get_T_world_frame(C.EE_FRAME)

        j0, tip = jaw_and_tip(0.0)
        on_jaw = np.linalg.inv(j0) @ np.r_[tip[:3, 3], 1.0]  # the fingertip point, attached to the moving jaw
        j1, _ = jaw_and_tip(0.6)  # opened
        u = (j1 @ on_jaw)[:3] - tip[:3, 3]
        u -= np.dot(u, tip[:3, 2]) * tip[:3, 2]
        ub = (self.T.R.T @ u)[:2]
        ub /= np.linalg.norm(ub)
        return tuple(np.asarray(xy, float) - (C.JAW_SHIFT_MM if shift is None else shift) * ub), ub

    def _tilt_check(self, q, label):
        T = self.ik.fk(q)
        tilt = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))
        if tilt > C.GRASP_MAX_TILT:
            raise Unreachable(f"{label}: grasp tilt {tilt:.0f} deg > {C.GRASP_MAX_TILT}")

    # ---------------------------------------------------------------- execution
    def _as_dicts(self, qs, gripper):
        return [{**dict(zip(C.ARM_JOINTS, map(float, q))), "gripper": gripper} for q in qs]

    def _follow(self, qs, gripper, duration, check=None):
        path = self._as_dicts(qs, gripper)
        self.arm.move_to(path[0])
        if len(path) > 1:
            return self.arm.move_path(path, duration, check)
        return None

    def _holding_check(self, held):
        """While carrying: the jaws closing further than the holding reading means the piece slipped out."""
        if not self.verify_grasp:
            return None

        def check():
            g = self.arm.present()["gripper"]
            return f"gripper closed to {g:.1f}% (was holding at {held:.1f}%)" if g < held - C.DROP_DETECT_PCT else None

        return check

    def _escape_up(self):
        """Straight up from wherever the arm stopped, to the safe height, jaws open."""
        q = np.array([self.arm.joints()[j] for j in C.ARM_JOINTS])
        p = self.T.R.T @ (self.ik.fk(q)[:3, 3] - self.T.t) * 1000
        p[:2] -= np.asarray(C.ARM_TARGET_OFFSET_MM)  # back to "where the planner meant", like every other point
        try:
            up = self._line(p[:2], max(p[2], 0.0), p[:2], C.Z_SAFE, q, 90.0)
            self._follow(up, C.GRIPPER_OPEN, C.DESCEND_S)
        except Unreachable:
            self.arm.move_to({"shoulder_lift": self.arm.joints()["shoulder_lift"] - 10}, 1.0)

    def _grasp(self, plan):
        cmd = max(0.0, plan["grip"] - C.GRIP_SQUEEZE)
        self.arm.gripper(cmd)
        self.arm.dwell()
        if not self.verify_grasp:
            return True, cmd
        reading = self.arm.present()["gripper"]
        if reading <= cmd + C.GRIP_EMPTY_TOL:
            return False, f"empty (gripper {reading:.1f}%)"
        if reading > plan["grip"] + C.GRIP_TOO_WIDE:
            return False, f"misgrip (gripper {reading:.1f}%, expected ~{plan['grip']:.0f}%)"
        return True, reading

    def execute(self, t: Transfer, retries=2, obstacles=None):
        plan = self.plan_transfer(t, obstacles)
        self.log(f"  transfer: {t.label} ({plan['info']['route']}, up to {plan['info']['z_travel_mm']} mm)")
        a = plan["approach"]
        opening = plan.get("open", C.GRIPPER_OPEN)
        if plan["info"].get("grasp") == "tight":
            self.log(f"  tight spot: grasp opening {opening}%, release +{plan['release_pct']}% (from the crowded demos)")
        for attempt in range(retries + 1):
            self._follow(a[:1], opening, C.MIN_MOVE_S)  # hover high over the square
            self.arm.gripper(opening)
            self.arm.settle()  # learn the sag here; the correction carries into the descent
            self._follow(a, opening, C.DESCEND_S + 0.8)  # down, sideways step, down to the base
            ok, info = self._grasp(plan)
            if ok:
                break
            self.log(f"  grasp attempt {attempt + 1} failed: {info}")
            self.arm.gripper(opening)
            self._follow(plan["abort_up"], opening, C.DESCEND_S)
        else:
            raise GraspFailed(f"{t.label}: {info}")
        held = self.arm.present()["gripper"] if self.verify_grasp else self.arm.joints()["gripper"]
        check = self._holding_check(held)
        for seg, dur in (("up", 0.6), ("travel", plan["travel_s"]), ("down2", 0.8)):
            why = self._follow(plan[seg], held, dur, check)
            if why:
                self.log(f"  piece dropped while moving ({why}): going up")
                self.arm.gripper(C.GRIPPER_OPEN)
                self._escape_up()
                raise PieceDropped(f"{t.label}: {why}")
        let_go = min(opening, held + plan.get("release_pct", C.RELEASE_OPEN_PCT))  # open just enough: neighbours may be close
        self.arm.gripper(let_go)
        self.arm.dwell()
        let_go = self._make_sure_released(let_go, opening)
        self._follow(plan["up2"], let_go, C.DESCEND_S)

    def _make_sure_released(self, let_go, cap):
        """If the jaws still press on the piece after letting go (gripper load high), open more before lifting,
        so the piece isn't carried away."""
        if not self.verify_grasp or not hasattr(self.arm, "gripper_load"):
            return let_go
        for _ in range(8):
            try:
                load, at = self.arm.gripper_load(), self.arm.present()["gripper"]
            except Exception:
                return let_go
            short = at < let_go - 0.8  # the jaws didn't get where they were told: the piece is in the way
            if (load < C.RELEASE_FREE_LOAD_PCT and not short) or let_go >= cap:
                break
            let_go = min(cap, let_go + C.RELEASE_STEP_PCT)
            self.log(f"  jaws still pressing (load {load:.0f}%, at {at:.1f}%): opening to {let_go:.1f}%")
            self.arm.gripper(let_go, 0.4)
            self.arm.dwell(0.2)
        return let_go

    def go_look(self):
        """Back to the look/home pose, gripper closed (GRIPPER_LOOK)."""
        self.arm.move_to({**look_pose(), "gripper": C.GRIPPER_LOOK}, 2.0)
        if hasattr(self.arm, "reset_residual"):
            self.arm.reset_residual()

    def check(self, board: chess.Board, move: chess.Move, slots: Slots):
        """Feasibility report without moving (like chess-player's chess_move.plan). Uses a copy of the slots."""
        import copy

        if move not in board.legal_moves:
            return dict(valid=False, reason=f"{move.uci()} is not legal here")
        sl = copy.deepcopy(slots)
        steps, ok_all = [], True
        for t, obst in obstacles_for(board, sl, plan_move(board, move, sl)):
            try:
                info = self.plan_transfer(t, obst)["info"]
                steps.append(dict(ok=True, **info))
            except Unreachable as e:  # includes Blocked
                ok_all = False
                steps.append(dict(ok=False, step=t.label, problem=type(e).__name__, details=str(e)))
        return dict(valid=True, uci=move.uci(), san=board.san(move), physical_ok=ok_all, steps=steps)

    def play(self, board: chess.Board, move: chess.Move, slots: Slots, recover=None):
        """Play `move`. `recover(transfer, error)` (optional) is called when a grab fails or a piece is dropped; it
        returns "done" (the piece is on its target anyway), new source xy (board mm) to retry from, or None (give
        up: the error is raised)."""
        transfers = list(obstacles_for(board, slots, plan_move(board, move, slots)))
        for t, obst in transfers:  # fail before moving if any part is unreachable or blocked
            self.plan_transfer(t, obst)
        for t, obst in transfers:
            for attempt in range(C.RECOVERY_ATTEMPTS + 1):
                try:
                    self.execute(t, obstacles=obst)
                    break
                except (GraspFailed, PieceDropped) as e:
                    if recover is None or attempt == C.RECOVERY_ATTEMPTS:
                        raise
                    where = recover(t, e)
                    if where == "done":
                        break
                    if where is None:
                        raise
                    t = Transfer(t.piece, tuple(where), t.dst, t.label + " (retry)")
                    obst = {xy: s for xy, s in obst.items() if np.hypot(xy[0] - where[0], xy[1] - where[1]) > 1}
        self.go_look()
        return [t for t, _ in transfers]
