
import tomllib
from typing import Any
from typing import TypedDict

import cv2
import keyboard
import mss
import numpy as np
import pytesseract
import pydirectinput


DEFAULT_CONFIG = '''
[capture]
# Id of the monitor to capture,
# just experiment with it until you get the right one
monitor_id = 3


[hotkey]
# Hotkey detection is using the keyboard PyPI package
# https://github.com/boppreh/keyboard
solve = "F13"
autoinput = "F14"
exit = "F15"


[bindings]
# Ingame key binding for rotating picks left
rotate_left = "a"
# Ingame key binding for rotating picks right
rotate_right = "d"
# Ingame key binding for selecting the previous pick
previous_pick = "q"
# Ingame key binding for selecting the next pick
next_pick = "t"
# Ingame key binding for selecting the pick
select_pick = "e"


[display]
# Window Title of the solved image window
title = "AutoDigipick"
# Window Resolution of the solved image window
resolution = [1920, 1080]
# Window Position of solved image window, offset from top left corner
# of the primary monitor
position = [1000, 2300]


[general]
# Number of circle segments, unless Bethesda does something wild, don't change this
num_segments = 32
# ID of the font to use for the display text
# FONT_HERSHEY_SIMPLEX = 0
# FONT_HERSHEY_PLAIN = 1
# FONT_HERSHEY_DUPLEX = 2
# FONT_HERSHEY_COMPLEX = 3
# FONT_HERSHEY_TRIPLEX = 4
# FONT_HERSHEY_COMPLEX_SMALL = 5
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6
# FONT_HERSHEY_SCRIPT_COMPLEX = 7
font = 2  # cv2.FONT_HERSHEY_DUPLEX
# Scale factor of the display text
font_scale = 1
# Thickness of the display text
thickness = 2
# Color rotation for digipicks used in solution (BGR)
# if the solution requires more picks than colors, the colors will be reused
# in order
colors = [
  [0, 0, 255],
  [0, 255, 255],
  [0, 255, 0],
  [255, 255, 0],
  [255, 0, 0],
  [255, 0, 255]
]


[security_level]
# Location of the security level text (top left corner)
# NOTE: AFTER normalization to 1920x1080
location = [1420, 250]
# Size of the security level text (bounding box)
# NOTE: AFTER normalization to 1920x1080
box_size = [200, 40]
# Lower threshold for masking only the security level text (BGR)
lower_threshold = [10, 50, 220]
# Upper threshold for masking only the security level text (BGR)
upper_threshold = [20, 70, 230]


[locks]
# Region size used to detect the presence/absence of circle segments
# NOTE: AFTER normalization to 1920x1080
box_size = [5, 5]
# Center of the lock circle (most likely the very center of the screen)
# NOTE: AFTER normalization to 1920x1080
center = [960, 540]
# Radius of the different lock circles
# NOTE: AFTER normalization to 1920x1080
bright_radii = [105, 133, 170, 208]
# Radius of the different lock circles offset by a small amount to detect the
# relative difference in brightness between the circle segments and the rest
# NOTE: AFTER normalization to 1920x1080
dark_radii = [90, 117, 151, 186]


[picks]
# Region size used to detect the presence/absence of circle segments
# NOTE: AFTER normalization to 1920x1080
box_size = [2, 2]
# Offset from the top left digipick to the next digipick (right)
# NOTE: AFTER normalization to 1920x1080
step_right = 112.5
# Offset from the top left digipick to the next digipick (down)
# NOTE: AFTER normalization to 1920x1080
step_down = 110.5
# Radius of the digipick circle
# NOTE: AFTER normalization to 1920x1080
radius = 39

[picks.novice]
# Center of the top left digipick
# NOTE: AFTER normalization to 1920x1080
center = [1405, 561]
# Number of digipicks available for the given difficulty
count = 4

[picks.advanced]
# Center of the top left digipick
# NOTE: AFTER normalization to 1920x1080
center = [1405, 511]
# Number of digipicks available for the given difficulty
count = 6

[picks.expert]
# Center of the top left digipick
# NOTE: AFTER normalization to 1920x1080
center = [1405, 460]
# Number of digipicks available for the given difficulty
count = 9

[picks.master]
# Center of the top left digipick
# NOTE: AFTER normalization to 1920x1080
center = [1405, 460]
# Number of digipicks available for the given difficulty
count = 12
'''


class CaptureConfig(TypedDict):
  monitor_id: int


class HotkeyConfig(TypedDict):
  solve: str
  autoinput: str
  exit: str


class BindingsConfig(TypedDict):
  rotate_left: str
  rotate_right: str
  previous_pick: str
  next_pick: str
  select_pick: str


class DisplayConfig(TypedDict):
  title: str
  resolution: tuple[int, int]
  position: tuple[int, int]


class GeneralConfig(TypedDict):
  num_segments: int
  font: int
  font_scale: int
  thickness: int
  colors: list[tuple[int, int, int]]


class SecurityLevelConfig(TypedDict):
  location: tuple[int, int]
  box_size: tuple[int, int]
  lower_threshold: tuple[int, int, int]
  upper_threshold: tuple[int, int, int]


class LocksConfig(TypedDict):
  box_size: tuple[int, int]
  center: tuple[int, int]
  bright_radii: tuple[float, ...]
  dark_radii: tuple[float, ...]


class PicksLevelConfig(TypedDict):
  center: tuple[int, int]
  count: int


class PicksConfig(TypedDict):
  box_size: tuple[int, int]
  step_right: float
  step_down: float
  radius: float
  novice: PicksLevelConfig
  advanced: PicksLevelConfig
  expert: PicksLevelConfig
  master: PicksLevelConfig


class ConfigDict(TypedDict):
  capture: CaptureConfig
  hotkey: HotkeyConfig
  bindings: BindingsConfig
  display: DisplayConfig
  general: GeneralConfig
  security_level: SecurityLevelConfig
  locks: LocksConfig
  picks: PicksConfig


def read_config(
  config_path: str = "autodigipick.toml",
  create_default: bool = True
) -> ConfigDict:
  try:
    with open(config_path, "rb") as config_file:
      config = tomllib.load(config_file)
  except FileNotFoundError:
    if create_default:
      with open(config_path, "wb") as config_file:
        config_file.write(DEFAULT_CONFIG.encode())
        config = tomllib.loads(DEFAULT_CONFIG)
  return config  # type: ignore


def try_digipick(
  lock: tuple[int, ...],
  digipick: tuple[int, ...]
) -> list[tuple[int, tuple[int, ...]]]:
  assert len(lock) == len(digipick)
  if all(
    digipick[i] == 0
    for i in range(len(digipick))
  ):
    return []
  results = []
  # rotate pick
  for i in range(len(digipick)):
    rotated_pick = digipick[i:] + digipick[:i]
    combined_lock = tuple(
      lock[j] + rotated_pick[j]
      for j in range(len(lock))
    )
    if all(
      combined_lock[j] <= 1
      for j in range(len(lock))
    ):
      results += [(i, combined_lock)]
  return results


def brute_force(
  locks: list[tuple[int, ...]],
  digipicks: list[tuple[int, ...]],
  solution_so_far: list[tuple[int, int, int]],
  lock_depth: int = 0,
) -> list[tuple[int, int, int]] | None:
  lock = locks[-1]
  USED_PICK = tuple(0 for _ in range(len(digipicks[0])))
  for i in range(len(digipicks)):
    results = try_digipick(lock, digipicks[i])
    if results:
      new_digipicks = digipicks[:i] + [USED_PICK] + digipicks[i+1:]
      for rotation, new_lock in results:
        solution = solution_so_far + [(i, rotation, lock_depth)]
        new_locks = locks[:-1]
        if not all(
          new_lock[j] == 1
          for j in range(len(new_lock))
        ):
          new_locks += [new_lock]
        else:
          lock_depth += 1
        if len(new_locks) == 0:
          return solution
        else:
          next_solution = brute_force(
            new_locks,
            new_digipicks,
            solution,
            lock_depth
          )
          if next_solution:
            return next_solution
  return None


def add_segments(
    scaled_image: Any,
    center_point: tuple[int, int],
    radius: float,
    segments: tuple[int, ...],
    dot_radius: int,
    color: tuple[int, int, int]
  ) -> None:
  num_segments = len(segments)

  # Define the number of segments and the angle step size
  angle_step = 2 * np.pi / num_segments

  # Create an array of angles for each segment
  angles = np.arange(0, 2 * np.pi, angle_step)

  # Calculate the x and y coordinates for each segment
  x_coords = center_point[0] + radius * np.cos(angles)
  y_coords = center_point[1] + radius * np.sin(angles)

  # Draw the segments
  for i, v in enumerate(segments):
    if v:
      cv2.circle(
        scaled_image,
        (int(x_coords[i]), int(y_coords[i])),
        dot_radius,
        color,
        -1
      )


def detect_security_level(
    image: Any,
    cfg: ConfigDict
  ):
  x = cfg["security_level"]["location"][0]
  y = cfg["security_level"]["location"][1]
  w = cfg["security_level"]["box_size"][0]
  h = cfg["security_level"]["box_size"][1]
  lower_red = tuple(cfg["security_level"]["lower_threshold"])
  upper_red = tuple(cfg["security_level"]["upper_threshold"])
  # Define the area of interest as a rectangle
  roi = image[y:y+h, x:x+w]
  modified_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
  modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)
  # Create a binary mask that selects only the pixels with red color values
  red_mask = cv2.inRange(modified_image, lower_red, upper_red)  # type: ignore
  # Apply the mask to the original image to filter out the non-red components
  masked_roi = cv2.bitwise_and(modified_image, modified_image, mask=red_mask)
  # Convert the ROI to grayscale
  gray_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
  # Apply thresholding to the grayscale ROI
  thresh_roi = cv2.threshold(
    gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )[1]
  # Use Tesseract to extract the text from the ROI
  return pytesseract.image_to_string(thresh_roi)


def solve_lock(cfg: ConfigDict, autoinput: bool = False):
  NUM_SEGMENTS = cfg["general"]["num_segments"]
  # Locks
  LOCK_SEGMENT_BOX = cfg["locks"]["box_size"]
  CENTER_POINT = cfg["locks"]["center"]
  BRIGHT_RADII = cfg["locks"]["bright_radii"]
  DARK_RADII = cfg["locks"]["dark_radii"]
  # Picks
  PICK_SEGMENT_BOX = cfg["picks"]["box_size"]
  PICK_STEP_RIGHT = cfg["picks"]["step_right"]
  PICK_STEP_DOWN = cfg["picks"]["step_down"]
  PICK_RADIUS = cfg["picks"]["radius"]

  FONT = cfg["general"]["font"]
  FONT_SCALE = cfg["general"]["font_scale"]
  THICKNESS = cfg["general"]["thickness"]
  COLORS = cfg["general"]["colors"]

  monitor_id = cfg["capture"]["monitor_id"]
  title = cfg["display"]["title"]
  window_resolution = cfg["display"]["resolution"]

  locks = []
  digipicks = []
  active_pick = 0
  potential_picks = 0

  with mss.mss() as sct:
    mon = sct.monitors[monitor_id]
    # The screen part to capture
    monitor = {
      "top": mon["top"],
      "left": mon["left"],
      "width": mon["width"],
      "height": mon["height"],
      "mon": monitor_id,
    }
    # Grab the data
    img = np.array(sct.grab(monitor)) # BGR Image

  scaled_image = cv2.resize(img, (1920, 1080)) # Normalize Resolution

  sec_level = detect_security_level(scaled_image, cfg).strip()
  print(f"Detected Security level: {sec_level}")
  if sec_level not in (
    "NOVICE",
    "ADVANCED",
    "EXPERT",
    "MASTER",
  ):
    print("Security level not detected!!!")
    solution = None
  else:
    bright_segments = []
    for radius in BRIGHT_RADII:
      # Define the number of segments and the angle step size
      angle_step = 2 * np.pi / NUM_SEGMENTS
      # Create an array of angles for each segment
      angles = np.arange(0, 2 * np.pi, angle_step)
      # Calculate the x and y coordinates for each segment
      x_coords = CENTER_POINT[0] + radius * np.cos(angles)
      y_coords = CENTER_POINT[1] + radius * np.sin(angles)

      def get_image_region(image, index):
        y_start = int(y_coords[index]) - LOCK_SEGMENT_BOX[1]
        y_end = int(y_coords[index]) + LOCK_SEGMENT_BOX[1]
        x_start = int(x_coords[index]) - LOCK_SEGMENT_BOX[0]
        x_end = int(x_coords[index]) + LOCK_SEGMENT_BOX[0]
        return image[y_start:y_end, x_start:x_end]

      segments = tuple(
        cv2.mean(
          cv2.cvtColor(get_image_region(scaled_image, i), cv2.COLOR_BGR2GRAY)
        )[0]
        for i in range(NUM_SEGMENTS)
      )
      bright_segments += [segments]

    dark_segments = []
    for radius in DARK_RADII:
      # Define the number of segments and the angle step size
      angle_step = 2 * np.pi / NUM_SEGMENTS
      # Create an array of angles for each segment
      angles = np.arange(0, 2 * np.pi, angle_step)
      # Calculate the x and y coordinates for each segment
      x_coords = CENTER_POINT[0] + radius * np.cos(angles)
      y_coords = CENTER_POINT[1] + radius * np.sin(angles)

      def get_image_region(image, index):
        y_start = int(y_coords[index]) - LOCK_SEGMENT_BOX[1]
        y_end = int(y_coords[index]) + LOCK_SEGMENT_BOX[1]
        x_start = int(x_coords[index]) - LOCK_SEGMENT_BOX[0]
        x_end = int(x_coords[index]) + LOCK_SEGMENT_BOX[0]
        return image[y_start:y_end, x_start:x_end]

      segments = tuple(
        cv2.mean(
          cv2.cvtColor(get_image_region(scaled_image, i), cv2.COLOR_BGR2GRAY)
        )[0]
        for i in range(NUM_SEGMENTS)
      )
      dark_segments += [segments]

    # print("Lock")
    for j in range(len(bright_segments)):
      delta = tuple(
        1 * ((bright_segments[j][i] - dark_segments[j][i]) > 5)
        for i in range(NUM_SEGMENTS)
      )
      # print(delta)
      locks += [delta]

    # cull non existant locks
    locks = [
      lock for lock in locks if any(lock[i] == 1 for i in range(len(lock)))
    ]

    if sec_level == "NOVICE":
      digipick_topleft = cfg["picks"]["novice"]["center"]
      potential_picks = cfg["picks"]["novice"]["count"]
    elif sec_level == "ADVANCED":
      digipick_topleft = cfg["picks"]["advanced"]["center"]
      potential_picks = cfg["picks"]["advanced"]["count"]
    elif sec_level == "EXPERT":
      digipick_topleft = cfg["picks"]["expert"]["center"]
      potential_picks = cfg["picks"]["expert"]["count"]
    else:
      digipick_topleft = cfg["picks"]["master"]["center"]
      potential_picks = cfg["picks"]["master"]["count"]

    pick_positions = [
      (
        int(digipick_topleft[0] + PICK_STEP_RIGHT * i),
        int(digipick_topleft[1] + PICK_STEP_DOWN * j),
      )
      for j in range(3)
      for i in range(4)
    ][:potential_picks]

    # print("Digipicks")
    for i, pick_pos in enumerate(pick_positions):
      # Define the number of segments and the angle step size
      angle_step = 2 * np.pi / NUM_SEGMENTS
      # Create an array of angles for each segment
      angles = np.arange(0, 2 * np.pi, angle_step)
      # Calculate the x and y coordinates for each segment
      x_coords = pick_pos[0] + PICK_RADIUS * np.cos(angles)
      y_coords = pick_pos[1] + PICK_RADIUS * np.sin(angles)

      def get_image_region(image, index):
        y_start = int(y_coords[index]) - PICK_SEGMENT_BOX[1]
        y_end = int(y_coords[index]) + PICK_SEGMENT_BOX[1]
        x_start = int(x_coords[index]) - PICK_SEGMENT_BOX[0]
        x_end = int(x_coords[index]) + PICK_SEGMENT_BOX[0]
        return image[y_start:y_end, x_start:x_end]

      segments = tuple(
        cv2.mean(
          cv2.cvtColor(get_image_region(scaled_image, i), cv2.COLOR_BGR2GRAY)
        )[0]
        for i in range(NUM_SEGMENTS)
      )

      def get_center_region(image):
        y_start = int(pick_pos[1]) - PICK_SEGMENT_BOX[1]
        y_end = int(pick_pos[1]) + PICK_SEGMENT_BOX[1]
        x_start = int(pick_pos[0]) - PICK_SEGMENT_BOX[0]
        x_end = int(pick_pos[0]) + PICK_SEGMENT_BOX[0]
        return image[y_start:y_end, x_start:x_end]

      center_brightness = (
        cv2.mean(
          cv2.cvtColor(get_center_region(scaled_image), cv2.COLOR_BGR2GRAY)
        )[0]
      )
      if center_brightness > 120:
        active_pick = i

      threshold = (max(segments) + min(segments)) / 2

      delta = tuple(
        (segments[i] > threshold) * 1
        for i in range(NUM_SEGMENTS)
      )

      digipicks += [delta]

    solution = brute_force(locks, digipicks, [], 0)

  if solution:
    for i, (pick_num, rotation, lock_depth) in enumerate(solution):
      print(f"Pick {pick_num+1} rotated {rotation} times")

      # Define the position of the text
      text_position = pick_positions[pick_num]
      text = f"{i+1}"
      color = COLORS[i % len(COLORS)]

      # Get the size of the text
      (text_width, text_height), _ = cv2.getTextSize(
        text, FONT, FONT_SCALE, THICKNESS
      )

      # Calculate the position of the text to center it on text_position
      text_x = int(text_position[0] - text_width / 2)
      text_y = int(text_position[1] + text_height / 2)

      # Write the text on the image in red color
      cv2.putText(
        scaled_image,
        text,
        (text_x, text_y),
        FONT,
        FONT_SCALE,
        color,
        THICKNESS
      )

      rotated_pick = (
        digipicks[pick_num][rotation:] + digipicks[pick_num][:rotation]
      )
      # Add Digipick display
      add_segments(
        scaled_image, text_position, PICK_RADIUS, rotated_pick, 3, color
      )
      # Add Lock display
      lock_radius = BRIGHT_RADII[-1 - lock_depth]
      add_segments(
        scaled_image, CENTER_POINT, lock_radius, rotated_pick, 10, color
      )
  else:
    print("No solution found")
    # Define the position of the text
    text_position = (1920 // 2, 1080 // 2)
    text = f"No solution found"

    if locks:
      print("Detected Locks:")
      color = (0, 255, 255)
      offset = len(BRIGHT_RADII) - len(locks)
      for i, lock in enumerate(locks):
        print(lock)
        lock_radius = BRIGHT_RADII[i + offset]
        add_segments(
          scaled_image, CENTER_POINT, lock_radius, lock, 5, color
        )

    if digipicks:
      color = (0, 255, 255)
      print("Detected Digipicks:")
      for i, digipick in enumerate(digipicks):
        print(digipick)
        add_segments(
          scaled_image, pick_positions[i], PICK_RADIUS, digipick, 3, color
        )

    # Get the size of the text
    (text_width, text_height), _ = cv2.getTextSize(
      text, FONT, 5 * FONT_SCALE, THICKNESS
    )

    # Calculate the position of the text to center it on text_position
    text_x = int(text_position[0] - text_width / 2)
    text_y = int(text_position[1] + text_height / 2)

    # Write the text on the image in red color
    cv2.putText(
      scaled_image,
      text,
      (text_x, text_y),
      FONT,
      5 * FONT_SCALE,
      (0, 0, 255),
      THICKNESS
    )

  # Display the picture
  scaled_image = cv2.resize(scaled_image, window_resolution)
  cv2.imshow(title, scaled_image)

  if autoinput and solution:
    cv2.waitKey(10)
    ROTATE_LEFT = cfg["bindings"]["rotate_left"]
    ROTATE_RIGHT = cfg["bindings"]["rotate_right"]
    PREVIOUS_PICK = cfg["bindings"]["previous_pick"]
    NEXT_PICK = cfg["bindings"]["next_pick"]
    SELECT_PICK = cfg["bindings"]["select_pick"]
    available_picks = [1 for _ in range(potential_picks)]
    for j, (pick_num, rotation, lock_depth) in enumerate(solution):
      if pick_num >= active_pick:
        for i in range(active_pick, pick_num, 1):
          if available_picks[i+1]:
            pydirectinput.press(NEXT_PICK, _pause=False, duration=0.05)
            cv2.waitKey(50)
      else:
        for i in range(active_pick, pick_num, -1):
          if available_picks[i-1]:
            pydirectinput.press(PREVIOUS_PICK, _pause=False, duration=0.05)
            cv2.waitKey(50)
      cv2.waitKey(50)
      if rotation > NUM_SEGMENTS // 2:
        rotation = NUM_SEGMENTS - rotation
        rotate = ROTATE_RIGHT
      else:
        rotate = ROTATE_LEFT
      for _ in range(rotation):
        pydirectinput.press(rotate, _pause=False, duration=0.05)
        cv2.waitKey(50)
      pydirectinput.press(SELECT_PICK, _pause=False, duration=0.05)
      cv2.waitKey(100)
      available_picks[pick_num] = 0
      # active_pick switches to the next pick after pick_num, skipping all used picks
      for i in range(potential_picks):
        pick = (pick_num + i) % potential_picks
        if available_picks[pick]:
          active_pick = pick
          break


def run_AutoDigipick(
  config_path: str = "autodigipick.toml"
) -> None:
  cfg = read_config(config_path)
  window_position = cfg["display"]["position"]
  title = cfg["display"]["title"]
  solve_key = cfg["hotkey"]["solve"]
  autoinput_key = cfg["hotkey"]["autoinput"]
  exit_key = cfg["hotkey"]["exit"]
  cv2.startWindowThread()
  # Create a named window
  cv2.namedWindow(title)
  # Move it to (x,y)
  cv2.moveWindow(title, *window_position)
  assert keyboard.parse_hotkey(solve_key) is not None
  assert keyboard.parse_hotkey(exit_key) is not None
  keep_running = True
  while keep_running:
    if keyboard.is_pressed(exit_key):
      keep_running = False
      return
    if keyboard.is_pressed(solve_key):
      solve_lock(cfg, autoinput=False)
      cv2.waitKey(10)
    if keyboard.is_pressed(autoinput_key):
      solve_lock(cfg, autoinput=True)
      cv2.waitKey(10)
    cv2.waitKey(10)
