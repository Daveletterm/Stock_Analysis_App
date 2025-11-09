"""Mythic Bond web backend.

Install dependencies: pip install -r requirements.txt
Run the web app with: flask --app rpg_app.py run
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------------
# Core data models
# ----------------------------------------------------------------------------


@dataclass
class Move:
    name: str
    power: int
    accuracy: float  # value between 0.0 and 1.0
    type: str
    category: str  # "physical", "special", or "status"
    effect: Optional[Dict[str, object]] = None


@dataclass
class Monster:
    name: str
    level: int
    max_hp: int
    current_hp: int
    attack: int
    defense: int
    sp_attack: int
    sp_defense: int
    speed: int
    types: Tuple[str, ...]
    moves: List[Move] = field(default_factory=list)
    exp: int = 0
    exp_to_next: int = 20
    stat_stages: Dict[str, int] = field(
        default_factory=lambda: {stat: 0 for stat in ("attack", "defense", "sp_attack", "sp_defense", "speed")}
    )
    last_damage_taken: int = 0
    last_damage_category: Optional[str] = None
    flinched: bool = False
    front_sprite: Optional[str] = None
    back_sprite: Optional[str] = None
    front_color: Optional[Tuple[int, int, int]] = None
    back_color: Optional[Tuple[int, int, int]] = None

    def is_fainted(self) -> bool:
        return self.current_hp <= 0

    def heal(self) -> None:
        self.current_hp = self.max_hp
        for stat in self.stat_stages:
            self.stat_stages[stat] = 0
        self.last_damage_taken = 0
        self.last_damage_category = None
        self.flinched = False

    def gain_experience(self, amount: int) -> List[str]:
        messages: List[str] = []
        self.exp += amount
        while self.exp >= self.exp_to_next:
            self.exp -= self.exp_to_next
            self.level += 1
            # Simple stat growth for prototype level ups
            self.max_hp += 3
            self.attack += 2
            self.defense += 2
            self.sp_attack += 2
            self.sp_defense += 2
            self.speed += 1
            self.current_hp = self.max_hp
            self.exp_to_next = max(20, int(self.exp_to_next * 1.3))
            messages.append(f"{self.name} grew to level {self.level}!")
        return messages

    def reset_battle_state(self) -> None:
        for stat in self.stat_stages:
            self.stat_stages[stat] = 0
        self.last_damage_taken = 0
        self.last_damage_category = None
        self.flinched = False

    def stage_multiplier(self, stat: str) -> float:
        stage = self.stat_stages.get(stat, 0)
        if stage >= 0:
            return (2 + stage) / 2
        return 2 / (2 - stage)

    def get_modified_stat(self, stat: str) -> int:
        base_value = getattr(self, stat)
        return max(1, int(base_value * self.stage_multiplier(stat)))

    def change_stage(self, stat: str, stages: int) -> int:
        if stat not in self.stat_stages:
            return 0
        original = self.stat_stages[stat]
        self.stat_stages[stat] = max(-6, min(6, self.stat_stages[stat] + stages))
        return self.stat_stages[stat] - original


SPRITE_FOLDER = Path("assets/sprites")
SPRITE_FOLDER.mkdir(parents=True, exist_ok=True)
SPRITE_SIZE = (96, 96)


def _sprite_asset(filename: str, fallback_color: Tuple[int, int, int]) -> Tuple[Optional[str], Tuple[int, int, int]]:
    path = SPRITE_FOLDER / filename
    if path.exists():
        return filename, fallback_color
    return None, fallback_color


# ----------------------------------------------------------------------------
# Moves and monster templates
# ----------------------------------------------------------------------------


def create_move_library() -> Dict[str, Move]:
    """Return the core move definitions. Extend this to add new moves."""
    return {
        "Cinder Snap": Move("Cinder Snap", power=40, accuracy=0.95, type="ember", category="physical"),
        "Leaf Gust": Move("Leaf Gust", power=40, accuracy=0.9, type="flora", category="special"),
        "Ripple Shot": Move("Ripple Shot", power=45, accuracy=0.9, type="aqua", category="special"),
        "Nuzzle": Move("Nuzzle", power=20, accuracy=1.0, type="normal", category="physical", effect={"flinch_chance": 0.1}),
        "Fairy Wind": Move("Fairy Wind", power=40, accuracy=1.0, type="fairy", category="special"),
        "Bubblebeam": Move(
            "Bubblebeam",
            power=65,
            accuracy=0.95,
            type="aqua",
            category="special",
            effect={"lower_stat": {"stat": "speed", "stages": 1, "chance": 0.2}},
        ),
        "Acid Armor": Move(
            "Acid Armor",
            power=0,
            accuracy=1.0,
            type="fairy",
            category="status",
            effect={"raise_stat": {"stat": "defense", "stages": 2}},
        ),
        "Paper Cut": Move(
            "Paper Cut",
            power=55,
            accuracy=0.95,
            type="fairy",
            category="physical",
            effect={"crit_bonus": True},
        ),
        "Swipe": Move("Swipe", power=20, accuracy=0.95, type="normal", category="physical", effect={"multi_hit": (2, 4)}),
        "Pursue": Move(
            "Pursue",
            power=50,
            accuracy=1.0,
            type="dark",
            category="physical",
            effect={"bonus_on_escape": 1.5},
        ),
        "Assist": Move("Assist", power=0, accuracy=1.0, type="normal", category="status", effect={"assist": True}),
        "Faint Attack": Move("Faint Attack", power=60, accuracy=1.0, type="dark", category="physical"),
        "Bite": Move("Bite", power=60, accuracy=1.0, type="dark", category="physical", effect={"flinch_chance": 0.3}),
        "Intimidate": Move(
            "Intimidate",
            power=0,
            accuracy=1.0,
            type="fighting",
            category="status",
            effect={"lower_stat": {"stat": "attack", "stages": 1, "chance": 1.0}},
        ),
        "Counter": Move("Counter", power=0, accuracy=1.0, type="fighting", category="status", effect={"counter": True}),
        "Rock Smash": Move(
            "Rock Smash",
            power=70,
            accuracy=0.9,
            type="fighting",
            category="physical",
            effect={"lower_stat": {"stat": "defense", "stages": 1, "chance": 0.5}},
        ),
        "Earth Fang": Move(
            "Earth Fang",
            power=80,
            accuracy=0.95,
            type="ground",
            category="physical",
            effect={"lower_stat": {"stat": "defense", "stages": 1, "chance": 0.3}},
        ),
        "Protect Pack": Move(
            "Protect Pack",
            power=0,
            accuracy=1.0,
            type="fighting",
            category="status",
            effect={"team_buff": {"stats": ("defense", "sp_defense"), "self_stages": 1, "ally_stages": 1}},
        ),
        "Howl of Valor": Move(
            "Howl of Valor",
            power=0,
            accuracy=1.0,
            type="fighting",
            category="status",
            effect={"team_buff": {"stats": ("attack",), "self_stages": 2, "ally_stages": 1}},
        ),
        "Seismic Pounce": Move(
            "Seismic Pounce",
            power=95,
            accuracy=0.9,
            type="fighting",
            category="physical",
            effect={"additional_types": ("ground",)},
        ),
    }


def create_monster_templates(move_library: Dict[str, Move]) -> Dict[str, Monster]:
    """Base monster data. Copy these templates when spawning monsters."""

    def make(front_name: str, back_name: Optional[str], fallback: Tuple[int, int, int]) -> Tuple[Optional[str], Optional[str], Tuple[int, int, int], Tuple[int, int, int]]:
        front, front_color = _sprite_asset(front_name, fallback)
        if back_name:
            back, back_color = _sprite_asset(back_name, fallback)
        else:
            back, back_color = front, front_color
        return front, back, front_color, back_color

    embercub_front, embercub_back, embercub_front_color, embercub_back_color = make(
        "embercub_front.png", "embercub_back.png", (255, 140, 90)
    )
    splashfin_front, splashfin_back, splashfin_front_color, splashfin_back_color = make(
        "splashfin_front.png", "splashfin_back.png", (90, 170, 255)
    )
    budling_front, budling_back, budling_front_color, budling_back_color = make(
        "budling_front.png", "budling_back.png", (120, 200, 90)
    )
    spraygit_front, spraygit_back, spraygit_front_color, spraygit_back_color = make(
        "spraygit_front.png", "spraygit_back.png", (180, 200, 255)
    )
    momo_front, momo_back, momo_front_color, momo_back_color = make("momo_front.png", None, (40, 40, 40))
    loki_front, loki_back, loki_front_color, loki_back_color = make("loki_front.png", None, (160, 120, 60))
    lokain_front, lokain_back, lokain_front_color, lokain_back_color = make("lokain_front.png", None, (180, 110, 40))

    return {
        "Embercub": Monster(
            name="Embercub",
            level=5,
            max_hp=30,
            current_hp=30,
            attack=12,
            defense=8,
            sp_attack=13,
            sp_defense=8,
            speed=10,
            types=("ember",),
            moves=[move_library["Cinder Snap"], move_library["Nuzzle"]],
            exp=0,
            exp_to_next=20,
            front_sprite=embercub_front,
            back_sprite=embercub_back,
            front_color=embercub_front_color,
            back_color=embercub_back_color,
        ),
        "Splashfin": Monster(
            name="Splashfin",
            level=5,
            max_hp=32,
            current_hp=32,
            attack=11,
            defense=9,
            sp_attack=14,
            sp_defense=11,
            speed=9,
            types=("aqua",),
            moves=[move_library["Ripple Shot"], move_library["Nuzzle"]],
            exp=0,
            exp_to_next=20,
            front_sprite=splashfin_front,
            back_sprite=splashfin_back,
            front_color=splashfin_front_color,
            back_color=splashfin_back_color,
        ),
        "Budling": Monster(
            name="Budling",
            level=5,
            max_hp=28,
            current_hp=28,
            attack=10,
            defense=11,
            sp_attack=12,
            sp_defense=13,
            speed=8,
            types=("flora",),
            moves=[move_library["Leaf Gust"], move_library["Nuzzle"]],
            exp=0,
            exp_to_next=20,
            front_sprite=budling_front,
            back_sprite=budling_back,
            front_color=budling_front_color,
            back_color=budling_back_color,
        ),
        # Spraygit – gentle paper guardian with fluid magic
        "Spraygit": Monster(
            name="Spraygit",
            level=6,
            max_hp=34,
            current_hp=34,
            attack=11,
            defense=12,
            sp_attack=16,
            sp_defense=16,
            speed=11,
            types=("aqua", "fairy"),
            moves=[
                move_library["Fairy Wind"],
                move_library["Bubblebeam"],
                move_library["Acid Armor"],
                move_library["Paper Cut"],
            ],
            exp=0,
            exp_to_next=30,
            front_sprite=spraygit_front,
            back_sprite=spraygit_back,
            front_color=spraygit_front_color,
            back_color=spraygit_back_color,
        ),
        # Momo – swift monochrome cat with clever tricks
        "Momo": Monster(
            name="Momo",
            level=6,
            max_hp=30,
            current_hp=30,
            attack=15,
            defense=10,
            sp_attack=12,
            sp_defense=11,
            speed=17,
            types=("normal", "dark"),
            moves=[
                move_library["Swipe"],
                move_library["Pursue"],
                move_library["Assist"],
                move_library["Faint Attack"],
            ],
            exp=0,
            exp_to_next=30,
            front_sprite=momo_front,
            back_sprite=momo_back,
            front_color=momo_front_color,
            back_color=momo_back_color,
        ),
        # Loki – stalwart brown dog who protects the frontline
        "Loki": Monster(
            name="Loki",
            level=7,
            max_hp=38,
            current_hp=38,
            attack=16,
            defense=15,
            sp_attack=11,
            sp_defense=14,
            speed=13,
            types=("fighting", "ground"),
            moves=[
                move_library["Bite"],
                move_library["Intimidate"],
                move_library["Counter"],
                move_library["Rock Smash"],
            ],
            exp=0,
            exp_to_next=35,
            front_sprite=loki_front,
            back_sprite=loki_back,
            front_color=loki_front_color,
            back_color=loki_back_color,
        ),
        # Lokain – evolved guardian hound with heroic presence
        "Lokain": Monster(
            name="Lokain",
            level=9,
            max_hp=46,
            current_hp=46,
            attack=20,
            defense=18,
            sp_attack=14,
            sp_defense=17,
            speed=15,
            types=("fighting", "ground"),
            moves=[
                move_library["Earth Fang"],
                move_library["Protect Pack"],
                move_library["Howl of Valor"],
                move_library["Seismic Pounce"],
            ],
            exp=0,
            exp_to_next=40,
            front_sprite=lokain_front,
            back_sprite=lokain_back,
            front_color=lokain_front_color,
            back_color=lokain_back_color,
        ),
    }


def clone_monster(template: Monster) -> Monster:
    monster = Monster(
        name=template.name,
        level=template.level,
        max_hp=template.max_hp,
        current_hp=template.current_hp,
        attack=template.attack,
        defense=template.defense,
        sp_attack=template.sp_attack,
        sp_defense=template.sp_defense,
        speed=template.speed,
        types=template.types,
        moves=list(template.moves),
        exp=template.exp,
        exp_to_next=template.exp_to_next,
        front_sprite=template.front_sprite,
        back_sprite=template.back_sprite,
        front_color=template.front_color,
        back_color=template.back_color,
    )
    monster.reset_battle_state()
    return monster


# ----------------------------------------------------------------------------
# Map data and utility helpers
# ----------------------------------------------------------------------------


TILE_SIZE = 32
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
MAP_LAYOUT = [
    "####################",
    "#....GGGGGG....GG..#",
    "#..######....####..#",
    "#..#....#....#..#..#",
    "#..#....#....#..#..#",
    "#....CCCCCCCC......#",
    "#....C......C..GG..#",
    "#....C.DD..D.C.....#",
    "#....CCCCCCCC......#",
    "#..................#",
    "#.SSS......GGGG....#",
    "#.SHS......GGGG....#",
    "#.SDS......GGGG....#",
    "#..................#",
    "####################",
]
MAP_WIDTH = len(MAP_LAYOUT[0])
MAP_HEIGHT = len(MAP_LAYOUT)

TILE_TYPES = {
    "#": {"color": (70, 70, 70), "walkable": False, "name": "Wall"},
    ".": {"color": (200, 200, 160), "walkable": True, "name": "Ground"},
    "G": {"color": (120, 200, 120), "walkable": True, "name": "Grass"},
    "C": {"color": (150, 150, 180), "walkable": False, "name": "Great Hall"},
    "D": {"color": (230, 210, 150), "walkable": True, "name": "Door"},
    "S": {"color": (140, 140, 170), "walkable": False, "name": "House Wall"},
    "H": {"color": (170, 230, 200), "walkable": True, "name": "Healing Floor"},
}


class GameState:
    def __init__(
        self,
        player_x: int,
        player_y: int,
        party: List[Monster],
        wild_pool_names: List[str],
        overworld_message: Optional[str] = None,
        encounter_log: Optional[List[str]] = None,
        battle: Optional["BattleState"] = None,
    ):
        self.player_x = player_x
        self.player_y = player_y
        self.party = party
        self.wild_pool_names = wild_pool_names
        self.overworld_message = overworld_message
        self.encounter_log = encounter_log or []
        self.battle = battle

    def to_dict(self) -> Dict[str, object]:
        return {
            "player": {"x": self.player_x, "y": self.player_y},
            "party": [monster_to_dict(monster) for monster in self.party],
            "wild_pool_names": list(self.wild_pool_names),
            "overworld_message": self.overworld_message,
            "encounter_log": list(self.encounter_log),
            "battle": battle_to_dict(self.battle) if self.battle else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, object],
        move_library: Dict[str, Move],
        monster_templates: Dict[str, Monster],
    ) -> "GameState":
        player = data.get("player", {})
        party_data = data.get("party", [])
        party = [monster_from_dict(mon, move_library) for mon in party_data]
        wild_pool_names = list(data.get("wild_pool_names", list(monster_templates.keys())))
        battle_data = data.get("battle")
        battle = battle_from_dict(battle_data, move_library) if battle_data else None
        return cls(
            player_x=player.get("x", 2),
            player_y=player.get("y", 2),
            party=party or [clone_monster(monster_templates["Embercub"])],
            wild_pool_names=wild_pool_names,
            overworld_message=data.get("overworld_message"),
            encounter_log=list(data.get("encounter_log", [])),
            battle=battle,
        )


class BattleState:
    def __init__(
        self,
        player_monster: Monster,
        enemy_monster: Monster,
        player_party: List[Monster],
        menu_state: str = "action",
        log: Optional[List[str]] = None,
        ended: bool = False,
        player_victory: bool = False,
    ):
        self.player_monster = player_monster
        self.enemy_monster = enemy_monster
        self.player_party = player_party
        self.menu_state = menu_state
        self.log = log or []
        self.ended = ended
        self.player_victory = player_victory
        self.player_recent_escape_attempt = False
        self.enemy_recent_escape_attempt = False

    def append_messages(self, messages: Sequence[str]) -> None:
        for msg in messages:
            if msg:
                self.log.append(msg)
        self.log = self.log[-20:]


def tile_at(x: int, y: int) -> str:
    if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH:
        return MAP_LAYOUT[y][x]
    return "#"


def can_walk(x: int, y: int) -> bool:
    tile = tile_at(x, y)
    return TILE_TYPES.get(tile, TILE_TYPES["#"])["walkable"]


def on_grass(x: int, y: int) -> bool:
    return tile_at(x, y) == "G"


def encounter_chance() -> bool:
    return random.random() < 0.1


# ----------------------------------------------------------------------------
# Battle calculations and effects
# ----------------------------------------------------------------------------


TYPE_EFFECTIVENESS: Dict[str, Dict[str, float]] = {
    "ember": {"flora": 2.0, "aqua": 0.5, "fairy": 1.0},
    "aqua": {"ember": 2.0, "flora": 0.5, "ground": 2.0},
    "flora": {"aqua": 2.0, "ember": 0.5, "ground": 2.0},
    "normal": {"dark": 1.0, "fairy": 1.0},
    "fairy": {"dark": 2.0, "fighting": 2.0, "ember": 0.5},
    "dark": {"fairy": 0.5, "fighting": 0.5},
    "fighting": {"dark": 2.0, "fairy": 0.5, "normal": 2.0, "ground": 1.0},
    "ground": {"ember": 2.0, "aqua": 1.0, "fairy": 1.0, "fighting": 1.0},
}


def type_multiplier(move: Move, defender_types: Sequence[str]) -> float:
    move_types = [move.type]
    if move.effect and "additional_types" in move.effect:
        move_types.extend(move.effect["additional_types"])  # type: ignore[index]
    multiplier = 1.0
    for m_type in move_types:
        chart = TYPE_EFFECTIVENESS.get(m_type, {})
        for d_type in defender_types:
            multiplier *= chart.get(d_type, 1.0)
    return multiplier


def calculate_damage(attacker: Monster, defender: Monster, move: Move) -> Tuple[int, float]:
    if move.category == "status" or move.power <= 0:
        return 0, 1.0

    attack_stat_name = "attack" if move.category == "physical" else "sp_attack"
    defense_stat_name = "defense" if move.category == "physical" else "sp_defense"
    attack_stat = attacker.get_modified_stat(attack_stat_name)
    defense_stat = max(1, defender.get_modified_stat(defense_stat_name))

    base = ((2 * attacker.level / 5 + 2) * move.power * attack_stat / defense_stat) / 10 + 2
    stab = 1.5 if move.type in attacker.types else 1.0
    effectiveness = type_multiplier(move, defender.types)
    variance = random.uniform(0.85, 1.0)
    damage = int(base * stab * effectiveness * variance)
    return max(1, damage), effectiveness


def accuracy_check(move: Move) -> bool:
    return random.random() <= move.accuracy


def calculate_exp_gain(defeated: Monster) -> int:
    return 10 + defeated.level * 5


STAT_LABELS = {
    "attack": "Attack",
    "defense": "Defense",
    "sp_attack": "Sp. Attack",
    "sp_defense": "Sp. Defense",
    "speed": "Speed",
}


def stage_change_message(monster: Monster, stat: str, delta: int) -> Optional[str]:
    if delta == 0:
        if monster.stat_stages.get(stat, 0) >= 6:
            return f"{monster.name}'s {STAT_LABELS.get(stat, stat)} won't go higher!"
        if monster.stat_stages.get(stat, 0) <= -6:
            return f"{monster.name}'s {STAT_LABELS.get(stat, stat)} won't go lower!"
        return None
    if delta > 0:
        if delta >= 2:
            return f"{monster.name}'s {STAT_LABELS.get(stat, stat)} sharply rose!"
        return f"{monster.name}'s {STAT_LABELS.get(stat, stat)} rose!"
    if delta <= -2:
        return f"{monster.name}'s {STAT_LABELS.get(stat, stat)} harshly fell!"
    return f"{monster.name}'s {STAT_LABELS.get(stat, stat)} fell!"


def apply_move_effects(
    battle: BattleState,
    attacker: Monster,
    defender: Monster,
    move: Move,
    damage: int,
    is_player_attacker: bool,
) -> List[str]:
    messages: List[str] = []
    if not move.effect:
        return messages

    effect = move.effect

    if "flinch_chance" in effect and damage > 0:
        chance = effect["flinch_chance"]
        if random.random() < chance:
            defender.flinched = True
            messages.append(f"{defender.name} flinched!")

    if "lower_stat" in effect:
        details = effect["lower_stat"]
        chance = details.get("chance", 1.0)
        if random.random() <= chance:
            delta = defender.change_stage(details["stat"], -abs(details["stages"]))
            msg = stage_change_message(defender, details["stat"], delta)
            if msg:
                messages.append(msg)

    if "raise_stat" in effect:
        details = effect["raise_stat"]
        delta = attacker.change_stage(details["stat"], abs(details["stages"]))
        msg = stage_change_message(attacker, details["stat"], delta)
        if msg:
            messages.append(msg)

    if "raise_multiple" in effect:
        details = effect["raise_multiple"]
        for stat in details["stats"]:
            delta = attacker.change_stage(stat, abs(details["stages"]))
            msg = stage_change_message(attacker, stat, delta)
            if msg:
                messages.append(msg)

    if "team_buff" in effect:
        details = effect["team_buff"]
        stats = details["stats"]
        self_stages = details.get("self_stages", 0)
        ally_stages = details.get("ally_stages", 0)
        for stat in stats:
            if self_stages:
                delta = attacker.change_stage(stat, self_stages)
                msg = stage_change_message(attacker, stat, delta)
                if msg:
                    messages.append(msg)
            if ally_stages and is_player_attacker:
                for ally in battle.player_party:
                    if ally is attacker:
                        continue
                    delta = ally.change_stage(stat, ally_stages)
                    msg = stage_change_message(ally, stat, delta)
                    if msg:
                        messages.append(msg)

    if effect.get("assist"):
        ally_moves: List[Move] = []
        for ally in battle.player_party:
            if ally is attacker:
                continue
            ally_moves.extend([m for m in ally.moves if not (m.effect and m.effect.get("assist"))])
        if not ally_moves:
            ally_moves = [m for m in attacker.moves if m.name != move.name and not (m.effect and m.effect.get("assist"))]
        if not ally_moves:
            messages.append("But it failed!")
            return messages
        chosen = random.choice(ally_moves)
        messages.append(f"{attacker.name} called {chosen.name}!")
        damage_override, effectiveness = calculate_damage(attacker, defender, chosen)
        messages.extend(resolve_damage_and_effects(battle, attacker, defender, chosen, damage_override, effectiveness, is_player_attacker))
        return messages

    if effect.get("counter"):
        if attacker.last_damage_taken > 0 and attacker.last_damage_category == "physical":
            damage_override = attacker.last_damage_taken * 2
            attacker.last_damage_taken = 0
            attacker.last_damage_category = None
            messages.append(f"{attacker.name} retaliated fiercely!")
            damage = max(1, damage_override)
            defender.current_hp = max(0, defender.current_hp - damage)
            defender.last_damage_taken = damage
            defender.last_damage_category = "physical"
            messages.append(f"It dealt {damage} damage!")
        else:
            messages.append("But it failed!")
        return messages

    if effect.get("multi_hit") and damage > 0:
        min_hits, max_hits = effect["multi_hit"]
        hits = random.randint(min_hits, max_hits)
        total_damage = 0
        for _ in range(hits):
            hit_damage, effectiveness = calculate_damage(attacker, defender, move)
            if effect.get("crit_bonus"):
                hit_damage = int(hit_damage * 1.25)
            defender.current_hp = max(0, defender.current_hp - hit_damage)
            total_damage += hit_damage
            if defender.is_fainted():
                break
        defender.last_damage_taken = total_damage
        defender.last_damage_category = move.category
        messages.append(f"Hit {hits} times for {total_damage} damage!")
        return messages

    if effect.get("bonus_on_escape") and damage > 0:
        if defender.current_hp <= defender.max_hp * 0.5 or (is_player_attacker and battle.enemy_recent_escape_attempt):
            bonus = effect["bonus_on_escape"]
            base_damage = defender.last_damage_taken or damage
            extra = int(base_damage * (bonus - 1.0))
            defender.current_hp = max(0, defender.current_hp - extra)
            defender.last_damage_taken += extra
            messages.append("It punished the fleeing foe!")

    if effect.get("crit_bonus") and damage > 0:
        crit_damage = int(damage * 0.25)
        defender.current_hp = max(0, defender.current_hp - crit_damage)
        defender.last_damage_taken += crit_damage
        messages.append("A razor-sharp critical slash!")

    return messages


def resolve_damage_and_effects(
    battle: BattleState,
    attacker: Monster,
    defender: Monster,
    move: Move,
    damage: int,
    effectiveness: float,
    is_player_attacker: bool,
) -> List[str]:
    messages: List[str] = []

    dealt_damage = 0
    if move.category != "status" and move.power > 0 and not (move.effect and move.effect.get("multi_hit")):
        defender.current_hp = max(0, defender.current_hp - damage)
        defender.last_damage_taken = damage
        defender.last_damage_category = move.category
        dealt_damage = damage
        messages.append(f"It dealt {damage} damage!")

    if move.category != "status" and move.power > 0:
        if effectiveness > 1.0:
            messages.append("It's super effective!")
        elif effectiveness < 1.0:
            messages.append("It's not very effective...")

    effect_messages = apply_move_effects(battle, attacker, defender, move, dealt_damage or damage, is_player_attacker)
    messages.extend(effect_messages)
    return messages


# ----------------------------------------------------------------------------
# Game state management helpers
# ----------------------------------------------------------------------------


def new_game_state(move_library: Dict[str, Move], monster_templates: Dict[str, Monster]) -> GameState:
    starter = clone_monster(monster_templates["Embercub"])
    wild_pool_names = list(monster_templates.keys())
    return GameState(player_x=2, player_y=2, party=[starter], wild_pool_names=wild_pool_names)


def heal_party(party: List[Monster]) -> None:
    for monster in party:
        monster.heal()


def start_wild_battle(state: GameState, monster_templates: Dict[str, Monster]) -> BattleState:
    enemy_name = random.choice(state.wild_pool_names)
    enemy_template = monster_templates[enemy_name]
    enemy = clone_monster(enemy_template)
    player = clone_monster(state.party[0])
    battle_party = [clone_monster(mon) for mon in state.party]
    battle = BattleState(player_monster=player, enemy_monster=enemy, player_party=battle_party)
    battle.append_messages([f"A wild {enemy.name} appeared!"])
    return battle


def move_player(state: GameState, dx: int, dy: int, monster_templates: Dict[str, Monster]) -> None:
    new_x = state.player_x + dx
    new_y = state.player_y + dy
    if not can_walk(new_x, new_y):
        state.overworld_message = "You can't go that way."
        return

    state.player_x = new_x
    state.player_y = new_y
    tile_symbol = tile_at(new_x, new_y)

    if tile_symbol == "H":
        heal_party(state.party)
        state.overworld_message = "Your party feels refreshed at the healing house."
        state.encounter_log.append("Rested at the healing house.")
    elif tile_symbol == "G" and encounter_chance():
        state.overworld_message = None
        state.encounter_log.append("A wild creature challenged you!")
        state.battle = start_wild_battle(state, monster_templates)
    else:
        state.overworld_message = TILE_TYPES[tile_symbol]["name"]

    state.encounter_log = state.encounter_log[-10:]


def battle_choose_fight(battle: BattleState) -> None:
    battle.menu_state = "move"


def battle_cancel_move(battle: BattleState) -> None:
    battle.menu_state = "action"


def enemy_turn(battle: BattleState) -> List[str]:
    attacker = battle.enemy_monster
    defender = battle.player_monster
    messages: List[str] = []

    battle.enemy_recent_escape_attempt = False

    if attacker.flinched:
        attacker.flinched = False
        messages.append(f"Wild {attacker.name} flinched!")
        return messages

    move = random.choice(attacker.moves)
    if move.category != "status" and move.power > 0 and not accuracy_check(move):
        messages.append(f"Wild {attacker.name}'s {move.name} missed!")
    else:
        messages.append(f"Wild {attacker.name} used {move.name}!")
        damage, effectiveness = calculate_damage(attacker, defender, move)
        messages.extend(resolve_damage_and_effects(battle, attacker, defender, move, damage, effectiveness, False))
        if defender.is_fainted():
            messages.append(f"{defender.name} fainted!")
            battle.ended = True
            battle.player_victory = False
    return messages


def battle_use_move(battle: BattleState, move_index: int) -> None:
    if battle.menu_state != "move":
        return

    attacker = battle.player_monster
    defender = battle.enemy_monster
    moves_len = len(attacker.moves)
    if move_index < 0 or move_index >= moves_len:
        return
    move = attacker.moves[move_index]

    battle.player_recent_escape_attempt = False

    if attacker.flinched:
        attacker.flinched = False
        battle.append_messages([f"{attacker.name} flinched!"])
        battle.menu_state = "action"
        if not battle.ended:
            battle.append_messages(enemy_turn(battle))
        return

    if move.category != "status" and move.power > 0 and not accuracy_check(move):
        battle.append_messages([f"{attacker.name}'s {move.name} missed!"])
    else:
        battle.append_messages([f"{attacker.name} used {move.name}!"])
        damage, effectiveness = calculate_damage(attacker, defender, move)
        battle.append_messages(resolve_damage_and_effects(battle, attacker, defender, move, damage, effectiveness, True))
        if defender.is_fainted():
            exp_gain = calculate_exp_gain(defender)
            battle.append_messages([f"Wild {defender.name} fainted!", f"{attacker.name} gained {exp_gain} EXP!"])
            level_messages = attacker.gain_experience(exp_gain)
            battle.append_messages(level_messages)
            battle.ended = True
            battle.player_victory = True

    battle.menu_state = "action"

    if not battle.ended:
        battle.append_messages(enemy_turn(battle))


def battle_attempt_escape(battle: BattleState) -> None:
    battle.player_recent_escape_attempt = True
    if random.random() < 0.5:
        battle.append_messages(["Got away safely!"])
        battle.ended = True
        battle.player_victory = False
    else:
        battle.append_messages(["Couldn't escape!"])
        if not battle.ended:
            battle.append_messages(enemy_turn(battle))


def finalize_battle(state: GameState) -> None:
    if not state.battle:
        return

    battle = state.battle
    state.party[0] = battle.player_monster
    if battle.player_victory:
        state.overworld_message = f"You defeated {battle.enemy_monster.name}!"
        state.encounter_log.append(f"Won against {battle.enemy_monster.name}.")
    else:
        heal_party(state.party)
        state.overworld_message = "Your party was healed after the tough battle."
        state.encounter_log.append("Took time to recover after a defeat.")
    state.encounter_log = state.encounter_log[-10:]
    state.battle = None


# ----------------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------------


def monster_to_dict(monster: Monster) -> Dict[str, object]:
    return {
        "name": monster.name,
        "level": monster.level,
        "max_hp": monster.max_hp,
        "current_hp": monster.current_hp,
        "attack": monster.attack,
        "defense": monster.defense,
        "sp_attack": monster.sp_attack,
        "sp_defense": monster.sp_defense,
        "speed": monster.speed,
        "types": list(monster.types),
        "moves": [move.name for move in monster.moves],
        "exp": monster.exp,
        "exp_to_next": monster.exp_to_next,
        "stat_stages": dict(monster.stat_stages),
        "last_damage_taken": monster.last_damage_taken,
        "last_damage_category": monster.last_damage_category,
        "flinched": monster.flinched,
        "front_sprite": monster.front_sprite,
        "back_sprite": monster.back_sprite,
        "front_color": list(monster.front_color) if monster.front_color else None,
        "back_color": list(monster.back_color) if monster.back_color else None,
    }


def monster_from_dict(data: Dict[str, object], move_library: Dict[str, Move]) -> Monster:
    moves = [move_library[name] for name in data.get("moves", []) if name in move_library]
    monster = Monster(
        name=data.get("name", "Unknown"),
        level=data.get("level", 1),
        max_hp=data.get("max_hp", 20),
        current_hp=data.get("current_hp", 20),
        attack=data.get("attack", 5),
        defense=data.get("defense", 5),
        sp_attack=data.get("sp_attack", 5),
        sp_defense=data.get("sp_defense", 5),
        speed=data.get("speed", 5),
        types=tuple(data.get("types", [])),
        moves=moves,
        exp=data.get("exp", 0),
        exp_to_next=data.get("exp_to_next", 20),
        front_sprite=data.get("front_sprite"),
        back_sprite=data.get("back_sprite"),
        front_color=tuple(data.get("front_color", [])) if data.get("front_color") else None,
        back_color=tuple(data.get("back_color", [])) if data.get("back_color") else None,
    )
    monster.stat_stages.update(data.get("stat_stages", {}))
    monster.last_damage_taken = data.get("last_damage_taken", 0)
    monster.last_damage_category = data.get("last_damage_category")
    monster.flinched = data.get("flinched", False)
    return monster


def battle_to_dict(battle: Optional[BattleState]) -> Optional[Dict[str, object]]:
    if not battle:
        return None
    return {
        "player_monster": monster_to_dict(battle.player_monster),
        "enemy_monster": monster_to_dict(battle.enemy_monster),
        "player_party": [monster_to_dict(mon) for mon in battle.player_party],
        "menu_state": battle.menu_state,
        "log": list(battle.log),
        "ended": battle.ended,
        "player_victory": battle.player_victory,
        "player_recent_escape_attempt": battle.player_recent_escape_attempt,
        "enemy_recent_escape_attempt": battle.enemy_recent_escape_attempt,
    }


def battle_from_dict(data: Dict[str, object], move_library: Dict[str, Move]) -> BattleState:
    player_monster = monster_from_dict(data.get("player_monster", {}), move_library)
    enemy_monster = monster_from_dict(data.get("enemy_monster", {}), move_library)
    player_party = [monster_from_dict(mon, move_library) for mon in data.get("player_party", [])]
    battle = BattleState(
        player_monster=player_monster,
        enemy_monster=enemy_monster,
        player_party=player_party or [player_monster],
        menu_state=data.get("menu_state", "action"),
        log=list(data.get("log", [])),
        ended=data.get("ended", False),
        player_victory=data.get("player_victory", False),
    )
    battle.player_recent_escape_attempt = data.get("player_recent_escape_attempt", False)
    battle.enemy_recent_escape_attempt = data.get("enemy_recent_escape_attempt", False)
    return battle


SAVE_PATH = Path("saves/savegame.json")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_game(state: GameState) -> None:
    if state.battle and not state.battle.ended:
        raise RuntimeError("Cannot save during an active battle.")
    data = state.to_dict()
    data["battle"] = None
    SAVE_PATH.write_text(json.dumps(data, indent=2))


def load_game(
    move_library: Dict[str, Move],
    monster_templates: Dict[str, Monster],
) -> GameState:
    if not SAVE_PATH.exists():
        raise FileNotFoundError("No save data found.")
    data = json.loads(SAVE_PATH.read_text())
    return GameState.from_dict(data, move_library, monster_templates)


__all__ = [
    "Move",
    "Monster",
    "GameState",
    "BattleState",
    "TILE_TYPES",
    "MAP_LAYOUT",
    "SPRITE_FOLDER",
    "create_move_library",
    "create_monster_templates",
    "clone_monster",
    "new_game_state",
    "move_player",
    "battle_choose_fight",
    "battle_cancel_move",
    "battle_use_move",
    "battle_attempt_escape",
    "finalize_battle",
    "save_game",
    "load_game",
    "battle_to_dict",
    "battle_from_dict",
    "monster_to_dict",
    "monster_from_dict",
]
