# -*- coding: utf-8 -*-
"""Dungeons -- a small state-based dungeon engine.

The world is described by an explicit GameState (rooms and their
connections, monsters, doors).  Every mutation goes through
``apply_action`` which is transactional: if an action fails, the state
is rolled back completely, so a failed action never pollutes the state
seen by later actions.  Save/load uses deep-copied plain-data snapshots,
so an opened door stays open and a dead monster stays dead after a
reload, and neither the live state nor the save data can pollute each
other.  All derived output iterates in sorted order, so results are
deterministic and independent of traversal/insertion order.

Run ``python dungeons.py`` to execute the table-driven test suite and a
short scripted demo.
"""
import copy
import sys
from dataclasses import dataclass, field

import ascii_images


class ActionError(Exception):
    """An action could not be applied; the state was rolled back."""


@dataclass
class Room:
    name: str
    connections: set = field(default_factory=set)


@dataclass
class Monster:
    name: str
    room: str
    alive: bool = True


@dataclass
class Door:
    name: str
    rooms: tuple
    locked: bool = True
    open: bool = False


@dataclass
class GameState:
    rooms: dict = field(default_factory=dict)
    monsters: dict = field(default_factory=dict)
    doors: dict = field(default_factory=dict)
    player: str = ""


def build_world(reverse=False):
    """Build the demo dungeon.

    ``reverse=True`` inserts every entity in the opposite order; the
    resulting state must behave and describe identically.
    """
    state = GameState()
    rooms = ["cell", "alley", "hall", "kitchen", "library"]
    doors = [
        Door("cell_door", ("cell", "alley")),
        Door("hall_door", ("alley", "hall")),
        Door("kitchen_door", ("hall", "kitchen")),
    ]
    monsters = [
        Monster("rat", "cell"),
        Monster("dragon", "alley"),
        Monster("spider", "kitchen"),
    ]
    links = [("hall", "library")]
    if reverse:
        rooms, doors, monsters, links = (
            rooms[::-1], doors[::-1], monsters[::-1], links[::-1])
    for name in rooms:
        state.rooms[name] = Room(name)
    for door in doors:
        state.doors[door.name] = door
    for monster in monsters:
        state.monsters[monster.name] = monster
    for a, b in links:
        state.rooms[a].connections.add(b)
        state.rooms[b].connections.add(a)
    state.player = "cell"
    return state


def _require(mapping, key, kind):
    try:
        return mapping[key]
    except KeyError:
        raise ActionError("unknown %s: %r" % (kind, key))


def _apply(state, action):
    kind = action[0]
    if kind == "unlock_door":
        door = _require(state.doors, action[1], "door")
        if not door.locked:
            raise ActionError("door %r is not locked" % door.name)
        door.locked = False
    elif kind == "open_door":
        door = _require(state.doors, action[1], "door")
        if door.locked:
            raise ActionError("door %r is locked" % door.name)
        if door.open:
            raise ActionError("door %r is already open" % door.name)
        door.open = True
    elif kind == "close_door":
        door = _require(state.doors, action[1], "door")
        if not door.open:
            raise ActionError("door %r is not open" % door.name)
        door.open = False
    elif kind == "kill_monster":
        monster = _require(state.monsters, action[1], "monster")
        if not monster.alive:
            raise ActionError("monster %r is already dead" % monster.name)
        monster.alive = False
    elif kind == "connect":
        a, b = action[1], action[2]
        room_a = _require(state.rooms, a, "room")
        if a == b:
            raise ActionError("cannot connect a room to itself")
        room_a.connections.add(b)  # mutates before the check below on purpose
        room_b = state.rooms.get(b)
        if room_b is None:
            raise ActionError("unknown room: %r" % b)  # rollback must undo the add
        room_b.connections.add(a)
    elif kind == "move":
        target = _require(state.rooms, action[1], "room")
        if target.name not in reachable_rooms(state, state.player):
            raise ActionError("no open path from %r to %r" % (state.player, target.name))
        state.player = target.name
    else:
        raise ActionError("unknown action: %r" % (kind,))


def apply_action(state, action):
    """Apply ``action`` transactionally.

    On failure the state is restored from a deep-copy snapshot, so a
    failed action leaves no trace and cannot pollute later actions.
    """
    snapshot = copy.deepcopy(state)
    try:
        _apply(state, action)
    except ActionError:
        state.rooms = snapshot.rooms
        state.monsters = snapshot.monsters
        state.doors = snapshot.doors
        state.player = snapshot.player
        raise


def reachable_rooms(state, room):
    """Rooms reachable from ``room`` via connections or open doors, sorted."""
    room = _require(state.rooms, room, "room")
    result = set(room.connections)
    for door in state.doors.values():
        if door.open and room.name in door.rooms:
            result.update(door.rooms)
    result.discard(room.name)
    return sorted(result)


def describe(state):
    """Deterministic textual snapshot; independent of insertion order."""
    lines = ["player: %s" % state.player]
    for name in sorted(state.rooms):
        room = state.rooms[name]
        lines.append("room %s -> %s" % (name, ",".join(sorted(room.connections)) or "-"))
    for name in sorted(state.doors):
        door = state.doors[name]
        status = "open" if door.open else ("locked" if door.locked else "closed")
        lines.append("door %s %s=%s" % (name, "-".join(sorted(door.rooms)), status))
    for name in sorted(state.monsters):
        monster = state.monsters[name]
        lines.append("monster %s@%s=%s" % (name, monster.room,
                                           "alive" if monster.alive else "dead"))
    return "\n".join(lines)


def save_game(state):
    """Snapshot the state as plain, deep-copied, deterministically ordered data."""
    return {
        "player": state.player,
        "rooms": {name: sorted(state.rooms[name].connections)
                  for name in sorted(state.rooms)},
        "monsters": {name: {"room": m.room, "alive": m.alive}
                     for name, m in sorted(state.monsters.items())},
        "doors": {name: {"rooms": sorted(d.rooms), "locked": d.locked, "open": d.open}
                  for name, d in sorted(state.doors.items())},
    }


def load_game(data):
    """Rebuild a state from save data.

    The input is deep-copied first, so later mutations of ``data`` (or of
    the returned state) cannot leak into each other.  Door and monster
    flags are restored exactly as saved -- an open door stays open.
    """
    data = copy.deepcopy(data)
    state = GameState(player=data["player"])
    for name, connections in data["rooms"].items():
        state.rooms[name] = Room(name, set(connections))
    for name, info in data["monsters"].items():
        state.monsters[name] = Monster(name, info["room"], info["alive"])
    for name, info in data["doors"].items():
        state.doors[name] = Door(name, tuple(info["rooms"]), info["locked"], info["open"])
    return state


# ---------------------------------------------------------------- tests

def _expect_failure(state, action):
    """Assert ``action`` fails and the state survives it byte-for-byte."""
    snapshot = copy.deepcopy(state)
    try:
        apply_action(state, action)
    except ActionError:
        pass
    else:
        raise AssertionError("action %r should have failed" % (action,))
    assert state == snapshot, "failed action %r polluted the state" % (action,)
    return snapshot


def check_unlock_door(state):
    apply_action(state, ("unlock_door", "cell_door"))
    assert state.doors["cell_door"].locked is False
    assert state.doors["cell_door"].open is False


def check_open_unlocked_door(state):
    apply_action(state, ("unlock_door", "cell_door"))
    apply_action(state, ("open_door", "cell_door"))
    assert state.doors["cell_door"].open is True


def check_open_locked_door_rolls_back(state):
    before = _expect_failure(state, ("open_door", "cell_door"))
    assert state.doors["cell_door"].locked is True
    assert state.doors["cell_door"].open is False
    assert state == before


def check_double_open_rolls_back(state):
    apply_action(state, ("unlock_door", "cell_door"))
    apply_action(state, ("open_door", "cell_door"))
    _expect_failure(state, ("open_door", "cell_door"))
    assert state.doors["cell_door"].open is True


def check_close_open_door(state):
    apply_action(state, ("unlock_door", "cell_door"))
    apply_action(state, ("open_door", "cell_door"))
    apply_action(state, ("close_door", "cell_door"))
    assert state.doors["cell_door"].open is False


def check_close_closed_door_rolls_back(state):
    _expect_failure(state, ("close_door", "cell_door"))
    assert state.doors["cell_door"].open is False


def check_kill_monster(state):
    apply_action(state, ("kill_monster", "dragon"))
    assert state.monsters["dragon"].alive is False
    assert state.monsters["rat"].alive is True  # unrelated monster untouched


def check_kill_dead_monster_rolls_back(state):
    apply_action(state, ("kill_monster", "dragon"))
    _expect_failure(state, ("kill_monster", "dragon"))
    assert state.monsters["dragon"].alive is False


def check_unknown_monster_rolls_back(state):
    _expect_failure(state, ("kill_monster", "ghost"))
    assert set(state.monsters) == {"rat", "dragon", "spider"}


def check_connect_rooms(state):
    apply_action(state, ("connect", "cell", "library"))
    assert "library" in state.rooms["cell"].connections
    assert "cell" in state.rooms["library"].connections


def check_connect_missing_room_rolls_back(state):
    _expect_failure(state, ("connect", "cell", "nowhere"))
    assert "nowhere" not in state.rooms["cell"].connections
    assert state.rooms["cell"].connections == set()


def check_move_through_connection(state):
    apply_action(state, ("connect", "cell", "library"))
    apply_action(state, ("move", "library"))
    assert state.player == "library"


def check_move_through_closed_door_rolls_back(state):
    _expect_failure(state, ("move", "alley"))
    assert state.player == "cell"


def check_move_through_open_door(state):
    apply_action(state, ("unlock_door", "cell_door"))
    apply_action(state, ("open_door", "cell_door"))
    apply_action(state, ("move", "alley"))
    assert state.player == "alley"


def check_failure_does_not_pollute_later_actions(state):
    _expect_failure(state, ("open_door", "cell_door"))
    _expect_failure(state, ("kill_monster", "ghost"))
    _expect_failure(state, ("move", "alley"))
    apply_action(state, ("unlock_door", "cell_door"))
    apply_action(state, ("open_door", "cell_door"))
    apply_action(state, ("move", "alley"))
    assert state.player == "alley"
    assert state.doors["cell_door"].open is True


def check_save_load_keeps_door_open(state):
    apply_action(state, ("unlock_door", "cell_door"))
    apply_action(state, ("open_door", "cell_door"))
    loaded = load_game(save_game(state))
    assert loaded.doors["cell_door"].open is True   # regression: stays open
    assert loaded.doors["cell_door"].locked is False
    assert loaded.doors["hall_door"].locked is True


def check_save_load_keeps_monster_dead(state):
    apply_action(state, ("kill_monster", "dragon"))
    loaded = load_game(save_game(state))
    assert loaded.monsters["dragon"].alive is False
    assert loaded.monsters["rat"].alive is True


def check_save_data_is_deep_copied(state):
    apply_action(state, ("unlock_door", "cell_door"))
    apply_action(state, ("open_door", "cell_door"))
    data = save_game(state)
    loaded = load_game(data)
    loaded.doors["cell_door"].open = False      # mutate the loaded state
    loaded.monsters["rat"].alive = False
    assert data["doors"]["cell_door"]["open"] is True   # save data unpolluted
    assert data["monsters"]["rat"]["alive"] is True
    data["doors"]["cell_door"]["open"] = False  # tamper with the save data
    data["monsters"]["rat"]["alive"] = False
    assert state.doors["cell_door"].open is True        # live state unpolluted
    assert state.monsters["rat"].alive is True
    assert loaded.doors["cell_door"].open is False      # earlier copy unaffected


def check_describe_is_order_independent(state):
    other = build_world(reverse=True)
    assert describe(state) == describe(other)
    apply_action(state, ("connect", "cell", "library"))
    apply_action(other, ("connect", "cell", "library"))
    assert describe(state) == describe(other)


def check_reachable_is_sorted_and_order_independent(state):
    apply_action(state, ("connect", "cell", "library"))
    apply_action(state, ("connect", "cell", "hall"))
    other = build_world(reverse=True)
    apply_action(other, ("connect", "cell", "hall"))
    apply_action(other, ("connect", "cell", "library"))
    assert reachable_rooms(state, "cell") == ["hall", "library"]
    assert reachable_rooms(state, "cell") == reachable_rooms(other, "cell")


TESTS = [
    ("unlock door marks it unlocked", check_unlock_door),
    ("open unlocked door succeeds", check_open_unlocked_door),
    ("open locked door fails and rolls back", check_open_locked_door_rolls_back),
    ("double open fails and rolls back", check_double_open_rolls_back),
    ("close open door succeeds", check_close_open_door),
    ("close closed door fails and rolls back", check_close_closed_door_rolls_back),
    ("kill monster succeeds", check_kill_monster),
    ("kill dead monster fails and rolls back", check_kill_dead_monster_rolls_back),
    ("kill unknown monster fails and rolls back", check_unknown_monster_rolls_back),
    ("connect rooms links both ways", check_connect_rooms),
    ("connect to missing room rolls back partial write", check_connect_missing_room_rolls_back),
    ("move through connection succeeds", check_move_through_connection),
    ("move through closed door fails and rolls back", check_move_through_closed_door_rolls_back),
    ("move through open door succeeds", check_move_through_open_door),
    ("failures do not pollute later actions", check_failure_does_not_pollute_later_actions),
    ("save/load keeps opened door open", check_save_load_keeps_door_open),
    ("save/load keeps dead monster dead", check_save_load_keeps_monster_dead),
    ("save data and state are deep-copy isolated", check_save_data_is_deep_copied),
    ("describe is independent of insertion order", check_describe_is_order_independent),
    ("reachable rooms are sorted and order independent", check_reachable_is_sorted_and_order_independent),
]


def run_tests():
    failures = []
    for name, check in TESTS:
        try:
            check(build_world())
        except (AssertionError, ActionError) as exc:
            failures.append((name, exc))
            print("FAIL %s: %s" % (name, exc))
        else:
            print("PASS %s" % name)
    print("%d/%d tests passed" % (len(TESTS) - len(failures), len(TESTS)))
    return not failures


def demo():
    state = build_world()
    script = [
        ("kill_monster", "rat"),
        ("unlock_door", "cell_door"),
        ("open_door", "cell_door"),
        ("move", "alley"),
        ("kill_monster", "dragon"),
    ]
    for action in script:
        apply_action(state, action)
    print(describe(state))
    print()
    ascii_images.ascii_dragon()
    print("The dragon is slain; the cell door stands open. Saving and reloading...")
    loaded = load_game(save_game(state))
    assert describe(loaded) == describe(state)
    print("Reloaded state is identical -- open doors stay open, dead monsters stay dead.")


def main():
    ascii_images.ascii_castle()
    print("\t\tDUNGEONS -- state engine self-test\n")
    ok = run_tests()
    print()
    if ok:
        demo()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
