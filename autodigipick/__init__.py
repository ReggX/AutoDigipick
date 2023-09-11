
from typing import Any
from typing import TypeVar

import cv2
import keyboard
import mss
import numpy as np
import pytesseract


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
  solution_so_far: list[tuple[int, int]],
  lock_depth: int = 0,
) -> list[tuple[int, int]] | None:
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


Image = TypeVar("Image")

def add_segments(
    scaled_image: Image, 
    center_point: tuple[int, int],
    radius: int, 
    segments: tuple[int, ...], 
    dot_radius: int, 
    color: tuple[int, int, int]
  ) -> Image:
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
    x: int = 1420, 
    y: int = 250, 
    w: int = 200, 
    h: int = 40
  ):
  # Define the area of interest as a rectangle
  roi = image[y:y+h, x:x+w]
  modified_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
  modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)
  # Define the lower and upper bounds of the red color range
  lower_red = (10, 50, 220)
  upper_red = (20, 70, 230)
  # Create a binary mask that selects only the pixels with red color values
  red_mask = cv2.inRange(modified_image, lower_red, upper_red)
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


def solve_lock(
  monitor_id: int = 1,
  window_resolution: tuple[int, int] = (1920, 1080),
):
  NUM_SEGMENTS = 32
  # Locks
  LOCK_SEGMENT_BOX = (5, 5)
  CENTER_POINT = (1920 // 2, 1080 // 2)
  LOCK_RADII = (105, 133, 170, 208)
  CONTRAST_RADII = (90, 117, 151, 186)
  # Picks
  PICK_SEGMENT_BOX = (2, 2)
  PICK_STEP_RIGHT = 112.5
  PICK_STEP_DOWN = 110.5
  PICK_RADIUS = 39
  
  FONT = cv2.FONT_HERSHEY_DUPLEX
  FONT_SCALE = 1
  THICKNESS = 2
  COLORS = [
    (0, 0, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 255)
  ]
  
  locks = []
  digipicks = []

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

  sec_level = detect_security_level(scaled_image).strip()
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
    for radius in LOCK_RADII:
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
    for radius in CONTRAST_RADII:
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
      digipick_topleft = (1405, 561)
      potential_picks = 4
    elif sec_level == "ADVANCED":
      digipick_topleft = (1405, 511)
      potential_picks = 6
    elif sec_level == "EXPERT":
      digipick_topleft = (1405, 460)
      potential_picks = 9
    else:
      digipick_topleft = (1405, 460)
      potential_picks = 12


    pick_positions = [
      (
        int(digipick_topleft[0] + PICK_STEP_RIGHT * i),
        int(digipick_topleft[1] + PICK_STEP_DOWN * j),
      )
      for j in range(3)
      for i in range(4)
    ][:potential_picks]

    # print("Digipicks")
    for pick_pos in pick_positions: 
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
      # print(tuple(
      #   f"{segment:>4.1f}"
      #   for segment in segments
      # ))

      threshold = (max(segments) + min(segments)) / 2

      delta = tuple(
        (segments[i] > threshold) * 1
        for i in range(NUM_SEGMENTS)
      )

      digipicks += [delta]

    solution = brute_force(locks, digipicks, [], 0)
    # print(solution)


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
      lock_radius = LOCK_RADII[-1 - lock_depth]
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
      offset = len(LOCK_RADII) - len(locks)
      for i, lock in enumerate(locks):
        print(lock)
        lock_radius = LOCK_RADII[i + offset]
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
  cv2.imshow("AutoDigipick", scaled_image)


def run_AutoDigipick(
  monitor_id: int = 1,
  window_resolution: tuple[int, int] = (1920, 1080),
  window_position: tuple[int, int] = (0, 0),
  solve_key: str = "F13",
  exit_key: str = "F14",
):
  cv2.startWindowThread()
  cv2.namedWindow("AutoDigipick")  # Create a named window
  cv2.moveWindow("AutoDigipick", *window_position)  # Move it to (x,y)
  keep_running = True
  while keep_running:
    if keyboard.is_pressed(exit_key):
      keep_running = False
      return
    if keyboard.is_pressed(solve_key):
      solve_lock(monitor_id, window_resolution)
      cv2.waitKey(10)
    cv2.waitKey(10)

