# Install pygame with: pip install pygame
# Run the game with: python main.py

import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pygame


# ----------------------------------------------------------------------------
# Data definitions for moves and monsters
# ----------------------------------------------------------------------------


@dataclass
class Move:
    name: str
    power: int
    accuracy: float  # value between 0.0 and 1.0
    type: str


@dataclass
class Monster:
    name: str
    level: int
    max_hp: int
    current_hp: int
    attack: int
    defense: int
    speed: int
    type: str
    moves: List[Move] = field(default_factory=list)
    exp: int = 0
    exp_to_next: int = 20

    def is_fainted(self) -> bool:
        return self.current_hp <= 0

    def heal(self) -> None:
        self.current_hp = self.max_hp

    def gain_experience(self, amount: int) -> List[str]:
        """Add experience and return messages for any level ups."""
        messages: List[str] = []
        self.exp += amount
        while self.exp >= self.exp_to_next:
            self.exp -= self.exp_to_next
            self.level += 1
            # Simple stat growth for prototype level ups
            self.max_hp += 3
            self.attack += 2
            self.defense += 2
            self.speed += 1
            self.current_hp = self.max_hp
            self.exp_to_next = max(20, int(self.exp_to_next * 1.3))
            messages.append(f"{self.name} grew to level {self.level}!")
        return messages


# ----------------------------------------------------------------------------
# Monster templates and helper functions for creating parties
# ----------------------------------------------------------------------------


def create_move_library() -> Dict[str, Move]:
    """Return the core move definitions. Extend this to add new moves."""
    return {
        "Cinder Snap": Move("Cinder Snap", power=18, accuracy=0.95, type="ember"),
        "Leaf Gust": Move("Leaf Gust", power=16, accuracy=0.9, type="flora"),
        "Ripple Shot": Move("Ripple Shot", power=20, accuracy=0.85, type="aqua"),
        "Nuzzle": Move("Nuzzle", power=10, accuracy=1.0, type="normal"),
    }


def create_monster_templates(move_library: Dict[str, Move]) -> Dict[str, Monster]:
    """Base monster data. Copy these templates when spawning monsters."""
    return {
        "Embercub": Monster(
            name="Embercub",
            level=5,
            max_hp=30,
            current_hp=30,
            attack=12,
            defense=8,
            speed=10,
            type="ember",
            moves=[move_library["Cinder Snap"], move_library["Nuzzle"]],
            exp=0,
            exp_to_next=20,
        ),
        "Splashfin": Monster(
            name="Splashfin",
            level=5,
            max_hp=32,
            current_hp=32,
            attack=11,
            defense=9,
            speed=9,
            type="aqua",
            moves=[move_library["Ripple Shot"], move_library["Nuzzle"]],
            exp=0,
            exp_to_next=20,
        ),
        "Budling": Monster(
            name="Budling",
            level=5,
            max_hp=28,
            current_hp=28,
            attack=10,
            defense=11,
            speed=8,
            type="flora",
            moves=[move_library["Leaf Gust"], move_library["Nuzzle"]],
            exp=0,
            exp_to_next=20,
        ),
    }


def clone_monster(template: Monster) -> Monster:
    """Create a copy of a monster template so encounters do not share state."""
    return Monster(
        name=template.name,
        level=template.level,
        max_hp=template.max_hp,
        current_hp=template.current_hp,
        attack=template.attack,
        defense=template.defense,
        speed=template.speed,
        type=template.type,
        moves=list(template.moves),
        exp=template.exp,
        exp_to_next=template.exp_to_next,
    )


# ----------------------------------------------------------------------------
# Overworld definitions: map data, tiles, and player movement helpers
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


class Player:
    def __init__(self, tile_x: int, tile_y: int):
        self.tile_x = tile_x
        self.tile_y = tile_y

    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.tile_x * TILE_SIZE, self.tile_y * TILE_SIZE, TILE_SIZE, TILE_SIZE)


# ----------------------------------------------------------------------------
# Battle system helpers
# ----------------------------------------------------------------------------


class BattleState:
    def __init__(self, player_monster: Monster, enemy_monster: Monster):
        self.player_monster = player_monster
        self.enemy_monster = enemy_monster
        self.menu_state = "action"  # "action" or "move"
        self.action_index = 0
        self.move_index = 0
        self.message_queue: List[Dict[str, Optional[Callable[[], None]]]] = []
        self.pending_enemy_turn = False
        self.after_battle_callback: Optional[Callable[[], None]] = None
        self.ended = False

    def queue_message(self, text: str, callback: Optional[Callable[[], None]] = None) -> None:
        self.message_queue.append({"text": text, "callback": callback})

    def pop_message(self) -> Optional[Dict[str, Optional[Callable[[], None]]]]:
        if self.message_queue:
            return self.message_queue.pop(0)
        return None


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------


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
    # Adjust this probability to balance encounter frequency.
    return random.random() < 0.1


def calculate_damage(attacker: Monster, defender: Monster, move: Move) -> int:
    base = move.power + attacker.attack - int(defender.defense * 0.5)
    base = max(base, 1)
    variance = random.uniform(0.85, 1.0)
    return max(1, int(base * variance))


def accuracy_check(move: Move) -> bool:
    return random.random() <= move.accuracy


def calculate_exp_gain(defeated: Monster) -> int:
    return 10 + defeated.level * 5


def start_battle(player_monster: Monster, wild_monsters: List[Monster]) -> BattleState:
    enemy_template = random.choice(wild_monsters)
    enemy = clone_monster(enemy_template)
    player_mon = clone_monster(player_monster)
    return BattleState(player_monster=player_mon, enemy_monster=enemy)


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------


def draw_text(surface: pygame.Surface, text: str, position: tuple[int, int], font: pygame.font.Font, color=(10, 10, 10)) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, position)


def draw_overworld(screen: pygame.Surface, player: Player, font: pygame.font.Font, message: Optional[str]) -> None:
    for y, row in enumerate(MAP_LAYOUT):
        for x, tile in enumerate(row):
            tile_info = TILE_TYPES[tile]
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, tile_info["color"], rect)
            pygame.draw.rect(screen, (30, 30, 30), rect, 1)

    pygame.draw.rect(screen, (220, 60, 60), player.rect())
    hint = message or "Use arrow keys to explore. Walk on grass to find creatures!"
    draw_text(screen, hint, (10, WINDOW_HEIGHT - 25), font)


def draw_hp_bar(surface: pygame.Surface, font: pygame.font.Font, monster: Monster, position: tuple[int, int]) -> None:
    x, y = position
    bar_width = 200
    bar_height = 20
    hp_ratio = monster.current_hp / monster.max_hp
    pygame.draw.rect(surface, (0, 0, 0), (x, y, bar_width, bar_height), 2)
    pygame.draw.rect(surface, (200, 60, 60), (x + 2, y + 2, int((bar_width - 4) * hp_ratio), bar_height - 4))
    draw_text(surface, f"{monster.name} Lv{monster.level}", (x, y - 22), font)
    draw_text(surface, f"HP: {monster.current_hp}/{monster.max_hp}", (x + 6, y + 2), font)


def draw_battle(screen: pygame.Surface, battle: BattleState, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
    screen.fill((220, 220, 255))
    draw_hp_bar(screen, font, battle.player_monster, (40, 320))
    draw_hp_bar(screen, font, battle.enemy_monster, (360, 120))
    draw_text(
        screen,
        f"EXP: {battle.player_monster.exp}/{battle.player_monster.exp_to_next}",
        (40, 350),
        small_font,
    )

    # Basic placeholders for monsters
    pygame.draw.circle(screen, (255, 120, 80), (120, 260), 40)
    pygame.draw.circle(screen, (80, 180, 255), (500, 200), 40)

    # Draw battle menu area
    menu_rect = pygame.Rect(20, 360, 600, 100)
    pygame.draw.rect(screen, (245, 245, 245), menu_rect)
    pygame.draw.rect(screen, (0, 0, 0), menu_rect, 2)

    current_message = battle.message_queue[0]["text"] if battle.message_queue else None

    if current_message:
        draw_text(screen, current_message, (menu_rect.x + 12, menu_rect.y + 12), small_font)
    elif battle.menu_state == "action":
        options = ["Fight", "Run"]
        for idx, option in enumerate(options):
            prefix = "> " if idx == battle.action_index else "  "
            draw_text(screen, prefix + option, (menu_rect.x + 12, menu_rect.y + 12 + idx * 24), small_font)
    elif battle.menu_state == "move":
        for idx, move in enumerate(battle.player_monster.moves):
            prefix = "> " if idx == battle.move_index else "  "
            text = f"{prefix}{move.name} ({int(move.accuracy * 100)}% accuracy)"
            draw_text(screen, text, (menu_rect.x + 12, menu_rect.y + 12 + idx * 24), small_font)


# ----------------------------------------------------------------------------
# Battle flow control
# ----------------------------------------------------------------------------


def handle_battle_input(event: pygame.event.Event, battle: BattleState) -> None:
    if battle.message_queue:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_z):
            message = battle.pop_message()
            if message and message["callback"]:
                message["callback"]()
            # After message callbacks run, check whether enemy turn should start
            if not battle.message_queue and battle.pending_enemy_turn:
                battle.pending_enemy_turn = False
                execute_enemy_turn(battle)
            elif not battle.message_queue and battle.after_battle_callback:
                callback = battle.after_battle_callback
                battle.after_battle_callback = None
                callback()
        return

    if event.type != pygame.KEYDOWN:
        return

    if battle.menu_state == "action":
        if event.key in (pygame.K_UP, pygame.K_DOWN):
            battle.action_index = (battle.action_index + (1 if event.key == pygame.K_DOWN else -1)) % 2
        elif event.key in (pygame.K_RETURN, pygame.K_z, pygame.K_SPACE):
            if battle.action_index == 0:
                battle.menu_state = "move"
                battle.move_index = 0
            else:
                attempt_escape(battle)
    elif battle.menu_state == "move":
        moves_len = len(battle.player_monster.moves)
        if event.key == pygame.K_UP:
            battle.move_index = (battle.move_index - 1) % moves_len
        elif event.key == pygame.K_DOWN:
            battle.move_index = (battle.move_index + 1) % moves_len
        elif event.key == pygame.K_ESCAPE:
            battle.menu_state = "action"
        elif event.key in (pygame.K_RETURN, pygame.K_z, pygame.K_SPACE):
            selected_move = battle.player_monster.moves[battle.move_index]
            execute_player_turn(battle, selected_move)


def execute_player_turn(battle: BattleState, move: Move) -> None:
    attacker = battle.player_monster
    defender = battle.enemy_monster

    if not accuracy_check(move):
        battle.queue_message(f"{attacker.name}'s {move.name} missed!")
    else:
        damage = calculate_damage(attacker, defender, move)
        defender.current_hp = max(0, defender.current_hp - damage)
        battle.queue_message(f"{attacker.name} used {move.name}!")
        battle.queue_message(f"It dealt {damage} damage!")

        if defender.is_fainted():
            exp_gain = calculate_exp_gain(defender)

            def award_exp() -> None:
                level_messages = attacker.gain_experience(exp_gain)
                for message in level_messages:
                    battle.queue_message(message)
                battle.after_battle_callback = lambda: setattr(battle, "ended", True)

            battle.queue_message(f"Wild {defender.name} fainted!")
            battle.queue_message(f"{attacker.name} gained {exp_gain} EXP!", callback=award_exp)
            battle.pending_enemy_turn = False
            return

    battle.pending_enemy_turn = True
    battle.menu_state = "action"
    battle.action_index = 0


def execute_enemy_turn(battle: BattleState) -> None:
    attacker = battle.enemy_monster
    defender = battle.player_monster
    move = random.choice(attacker.moves)

    if not accuracy_check(move):
        battle.queue_message(f"Wild {attacker.name}'s {move.name} missed!")
    else:
        damage = calculate_damage(attacker, defender, move)
        defender.current_hp = max(0, defender.current_hp - damage)
        battle.queue_message(f"Wild {attacker.name} used {move.name}!")
        battle.queue_message(f"It dealt {damage} damage!")
        if defender.is_fainted():
            battle.queue_message(f"{defender.name} fainted!")
            battle.after_battle_callback = lambda: setattr(battle, "ended", True)


def attempt_escape(battle: BattleState) -> None:
    if random.random() < 0.5:
        battle.queue_message("Got away safely!", callback=lambda: setattr(battle, "ended", True))
    else:
        battle.queue_message("Couldn't escape!")
        battle.pending_enemy_turn = True


# ----------------------------------------------------------------------------
# Main game loop
# ----------------------------------------------------------------------------


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Mythic Bond Prototype")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)

    move_library = create_move_library()
    monster_templates = create_monster_templates(move_library)
    player_party = [clone_monster(monster_templates["Embercub"])]
    wild_pool = [monster_templates[name] for name in monster_templates]

    player = Player(tile_x=2, tile_y=2)
    game_mode = "overworld"
    active_battle: Optional[BattleState] = None
    overworld_message: Optional[str] = None
    overworld_message_timer = 0

    def end_battle() -> None:
        nonlocal game_mode, active_battle
        if not active_battle:
            return
        if active_battle.player_monster.is_fainted():
            # Simple healing logic for prototype
            for monster in player_party:
                monster.heal()
        else:
            player_party[0] = active_battle.player_monster
        game_mode = "overworld"
        active_battle = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif game_mode == "overworld" and event.type == pygame.KEYDOWN:
                dx, dy = 0, 0
                if event.key == pygame.K_UP:
                    dy = -1
                elif event.key == pygame.K_DOWN:
                    dy = 1
                elif event.key == pygame.K_LEFT:
                    dx = -1
                elif event.key == pygame.K_RIGHT:
                    dx = 1

                if dx or dy:
                    new_x = player.tile_x + dx
                    new_y = player.tile_y + dy
                    if can_walk(new_x, new_y):
                        player.tile_x = new_x
                        player.tile_y = new_y
                        tile_symbol = tile_at(new_x, new_y)
                        if tile_symbol == "H":
                            for monster in player_party:
                                monster.heal()
                            overworld_message = "Your party was restored at the roadside house!"
                            overworld_message_timer = 180
                        if tile_symbol == "G" and encounter_chance():
                            game_mode = "battle"
                            active_battle = start_battle(player_party[0], wild_pool)
                            overworld_message = None

            elif game_mode == "battle" and active_battle:
                handle_battle_input(event, active_battle)

        screen.fill((0, 0, 0))

        if game_mode == "overworld":
            draw_overworld(screen, player, font, overworld_message)
        elif game_mode == "battle" and active_battle:
            draw_battle(screen, active_battle, font, small_font)
            if getattr(active_battle, "ended", False) and not active_battle.message_queue and not active_battle.pending_enemy_turn:
                end_battle()

        if overworld_message_timer > 0:
            overworld_message_timer -= 1
            if overworld_message_timer == 0:
                overworld_message = None

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()


"""
Run the game with: python main.py
Add new monsters by editing create_monster_templates.
Add new moves inside create_move_library.
Edit MAP_LAYOUT and TILE_TYPES to build new areas.
"""
