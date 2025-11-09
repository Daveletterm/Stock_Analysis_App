import os
from typing import Any, Dict, List

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    send_from_directory,
)

from main import (
    MAP_LAYOUT,
    SPRITE_FOLDER,
    TILE_TYPES,
    GameState,
    battle_attempt_escape,
    battle_cancel_move,
    battle_choose_fight,
    battle_use_move,
    create_monster_templates,
    create_move_library,
    finalize_battle,
    load_game,
    move_player,
    new_game_state,
    save_game,
)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "mythic-bond-secret")

MOVE_LIBRARY = create_move_library()
MONSTER_TEMPLATES = create_monster_templates(MOVE_LIBRARY)


def _state_from_session() -> GameState:
    data = session.get("game_state")
    if data:
        return GameState.from_dict(data, MOVE_LIBRARY, MONSTER_TEMPLATES)
    return new_game_state(MOVE_LIBRARY, MONSTER_TEMPLATES)


def _store_state(state: GameState) -> None:
    session["game_state"] = state.to_dict()
    session.modified = True


def _map_rows(state: GameState) -> List[List[Dict[str, Any]]]:
    rows: List[List[Dict[str, Any]]] = []
    for y, row in enumerate(MAP_LAYOUT):
        row_data: List[Dict[str, Any]] = []
        for x, symbol in enumerate(row):
            tile = TILE_TYPES[symbol]
            row_data.append(
                {
                    "symbol": symbol,
                    "name": tile["name"],
                    "color": "rgb({},{},{})".format(*tile["color"]),
                    "walkable": tile["walkable"],
                    "is_player": state.player_x == x and state.player_y == y,
                }
            )
        rows.append(row_data)
    return rows


@app.route("/sprites/<path:filename>")
def serve_sprite(filename: str):
    return send_from_directory(SPRITE_FOLDER, filename)


@app.route("/", methods=["GET", "POST"])
def game() -> str:
    state = _state_from_session()

    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "move":
                direction = request.form.get("direction")
                offsets = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
                dx, dy = offsets.get(direction, (0, 0))
                move_player(state, dx, dy, MONSTER_TEMPLATES)
            elif action == "fight" and state.battle:
                battle_choose_fight(state.battle)
            elif action == "cancel_move" and state.battle:
                battle_cancel_move(state.battle)
            elif action == "run" and state.battle:
                battle_attempt_escape(state.battle)
            elif action == "choose_move" and state.battle:
                index = int(request.form.get("index", 0))
                battle_use_move(state.battle, index)
            elif action == "save":
                save_game(state)
                flash("Game saved successfully.", "success")
            elif action == "load":
                state = load_game(MOVE_LIBRARY, MONSTER_TEMPLATES)
                flash("Save file loaded.", "success")
            elif action == "reset":
                state = new_game_state(MOVE_LIBRARY, MONSTER_TEMPLATES)
                flash("Started a new adventure.", "info")
        except FileNotFoundError:
            flash("No save data found.", "warning")
        except RuntimeError as exc:
            flash(str(exc), "warning")

        if state.battle and state.battle.ended:
            finalize_battle(state)

        _store_state(state)
        return redirect(url_for("game"))

    map_rows = _map_rows(state)
    _store_state(state)
    return render_template(
        "rpg/game.html",
        state=state,
        map_rows=map_rows,
        tile_types=TILE_TYPES,
    )


if __name__ == "__main__":
    app.run(debug=True)
