import numpy as np

PHASE_TO_REWARD = {
    "bin-picking": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "coffee-push": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "coffee-pull": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "door-open": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "door-close": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "window-open": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "window-close": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "drawer-open": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "drawer-close": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "lever-pull": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "peg-insert-side": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "faucet-close": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "push": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "push-back": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "disassemble": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "sweep-into": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "hand-insert": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "one_leg": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 999: -1},
    "cabinet": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 999: -1},
    "pick_up_cup": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "phone_on_base": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "take_umbrella_out_of_umbrella_stand": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "put_rubbish_in_bin": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
    "stack_wine": {0: 1, 1: 2, 2: 3, 3: 4, 999: -1},
}

TASK_TO_PHASE = {
    "PickSingleYCB-v0": 4,
    "one_leg": 6,
    "cabinet": 11,
    "bin-picking": 3,
    "coffee-push": 3,
    "coffee-pull": 3,
    "door-open": 3,
    "door-close": 3,
    "window-open": 3,
    "window-close": 3,
    "drawer-open": 3,
    "drawer-close": 3,
    "lever-pull": 2,
    "peg-insert-side": 4,
    "faucet-close": 2,
    "push": 3,
    "push-back": 3,
    "disassemble": 3,
    "sweep-into": 3,
    "hand-insert": 3,
    "pick_up_cup": 3,
    "phone_on_base": 4,
    "take_umbrella_out_of_umbrella_stand": 3,
    "put_rubbish_in_bin": 3,
    "stack_wine": 3,
}

TASK_TO_MAX_EPISODE_STEPS = {
    "PickSingleYCB-v0": 200,
    "one_leg": 600,
    "cabinet": 1500,
    "bin-picking": 250,
    "coffee-push": 250,
    "coffee-pull": 250,
    "door-open": 250,
    "door-close": 250,
    "window-open": 250,
    "window-close": 250,
    "drawer-open": 250,
    "drawer-close": 250,
    "lever-pull": 250,
    "peg-insert-side": 250,
    "faucet-close": 250,
    "push": 250,
    "push-back": 250,
    "disassemble": 250,
    "sweep-into": 250,
    "hand-insert": 250,
    "pick_up_cup": 150,
    "phone_on_base": 150,
    "put_rubbish_in_bin": 150,
    "take_umbrella_out_of_umbrella_stand": 150,
    "stack_wine": 150,
}

SKILL_TO_CLASS = {
    "one_leg": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 100: 5, 101: 5, 102: 5, 103: 5, 104: 5},
    "cabinet": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 10,
        100: 11,
        101: 11,
        102: 11,
        103: 11,
        104: 11,
        105: 11,
        106: 11,
        107: 11,
        108: 11,
        109: 11,
        110: 11,
    },
    "coffee-push": {0: 0, 1: 1, 2: 2, 100: 3, 101: 3, 102: 3},
    "door-open": {0: 0, 1: 1, 2: 2, 100: 3, 101: 3, 102: 3},
    "window-open": {0: 0, 1: 1, 2: 2, 100: 3, 101: 3, 102: 3},
    "window-close": {0: 0, 1: 1, 2: 2, 100: 3, 101: 3, 102: 3},
    "drawer-open": {0: 0, 1: 1, 2: 2, 100: 3, 101: 3, 102: 3},
    "drawer-close": {0: 0, 1: 1, 2: 2, 100: 3, 101: 3, 102: 3},
}

CLASS_TO_PHASE = {
    "one_leg": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 999},
    "cabinet": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 999,
    },
    "coffee-push": {0: 0, 1: 1, 2: 2, 3: 999},
    "door-open": {0: 0, 1: 1, 2: 2, 3: 999},
    "window-open": {0: 0, 1: 1, 2: 2, 3: 999},
    "window-close": {0: 0, 1: 1, 2: 2, 3: 999},
    "drawer-open": {0: 0, 1: 1, 2: 2, 3: 999},
    "drawer-close": {0: 0, 1: 1, 2: 2, 3: 999},
}


def get_metaworld_instruct(task_name, skill_phase, output_type="one"):
    options = None
    if task_name == "coffee-push":
        if skill_phase == 0:
            options = [
                "a robot arm picking up the coffee cup.",
                # "pick up the coffee cup with robotic arm.",
                # "grasp the coffee mug using robot arm.",
                # "lift the coffee cup through robotic manipulation.",
                # "retrieve the coffee cup with robotic limbs.",
                # "maneuver the robot arm to hold the coffee cup.",
                # "use robotic arm to clutch the coffee cup.",
                # "engage the coffee cup with robot arm.",
                # "robotic arm to fetch the coffee cup.",
                # "seize the coffee cup with robotic grippers.",
                # "robotic appendages to lift the coffee cup.",
                # "employ robotic arm to grasp the coffee mug.",
                # "manipulate the robot arm to pick up the coffee cup.",
                # "robotic hands to secure the coffee cup.",
                # "get the coffee cup using robotic arm.",
                # "robotic arm operation to lift the coffee cup.",
                # "handle the coffee cup with robot arm.",
                # "robotic actuators to grab the coffee cup.",
                # "utilize robotic arm to retrieve the coffee cup.",
                # "command robotic arm to grasp the coffee cup.",
                # "direct robotic arm to collect the coffee cup.",
            ]
        elif skill_phase == 1:
            options = [
                "a robot arm moving the coffee cup to the red target point.",
                # "push the robot cup to the red point with the robotic arm.",
                # "move the cup to the red mark using a robotic arm.",
                # "drive the cup towards the red spot with a robotic arm.",
                # "nudge the cup to the red area using a robotic arm.",
                # "shove the cup to the red point with a robotic arm.",
                # "guide the cup to the red target with a robotic arm.",
                # "propel the cup to the red point using a robotic arm.",
                # "slide the cup towards the red mark with a robotic arm.",
                # "advance the cup to the red spot with a robotic arm.",
                # "maneuver the cup to the red point with a robotic arm.",
                # "use a robotic arm to push the cup to the red point.",
                # "a robotic arm to position the cup at the red point.",
                # "employ a robotic arm to move the cup to the red spot.",
                # "a robotic arm to shove the cup towards the red area.",
                # "control a robotic arm to guide the cup to the red point.",
                # "command a robotic arm to slide the cup to the red mark.",
                # "direct a robotic arm to advance the cup to the red spot.",
                # "operate a robotic arm to nudge the cup to the red point.",
                # "manipulate a robotic arm to move the cup to the red area.",
                # "with a robotic arm, push the cup to the designated red point.",
            ]
        elif skill_phase >= 2:
            options = ["a robot arm pushing the button on the coffee machine."]

    if task_name == "coffee-pull":
        if skill_phase == 0:
            options = ["a robot arm grabbing the coffee cup."]
        elif skill_phase == 1:
            options = ["a robot arm moving the coffee cup to the green target point."]
        elif skill_phase >= 2:
            options = ["a robot arm holding the cup near the green target point."]

    elif task_name == "bin-picking":
        if skill_phase == 0:
            options = ["a robot arm grabbing the green cube in the red box."]
        elif skill_phase == 1:
            options = ["a robot arm lifting the green cube from the red box and move it to the blue box."]
        elif skill_phase >= 2:
            options = [
                "a robot arm placing the green cube in the blue box.",
            ]

    elif task_name == "door-open":
        if skill_phase == 0:
            options = ["a robot arm grabbing the door handle."]
        elif skill_phase == 1:
            options = ["a robot arm opening a door to the green target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the door handle near the green target point after opening.",
            ]

    elif task_name == "door-close":
        if skill_phase == 0:
            options = ["a robot arm grabbing the door handle."]
        elif skill_phase == 1:
            options = ["a robot arm closing a door to the green target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the door handle near the green target point after closing.",
            ]

    elif task_name == "window-open":
        if skill_phase == 0:
            options = ["a robot arm grabbing the window handle."]
        elif skill_phase == 1:
            options = ["a robot arm opening a window from right to left."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the window handle after opening.",
            ]

    elif task_name == "window-close":
        if skill_phase == 0:
            options = ["a robot arm grabbing the window handle."]
        elif skill_phase == 1:
            options = ["a robot arm closing a window from left to right."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the window handle after closing.",
            ]

    elif task_name == "drawer-open":
        if skill_phase == 0:
            options = ["a robot arm grabbing the drawer handle."]
        elif skill_phase == 1:
            options = ["a robot arm opening a drawer to the green target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the drawer handle near the green target point after opening.",
            ]

    elif task_name == "drawer-close":
        if skill_phase == 0:
            options = ["a robot arm grabbing the drawer handle."]
        elif skill_phase == 1:
            options = ["a robot arm opening a drawer to the green target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the drawer handle near the green target point after closing.",
            ]

    elif task_name == "lever-pull":
        if skill_phase == 0:
            options = ["a robot arm touching the lever."]
        elif skill_phase >= 1:
            options = ["a robot arm pulling up the lever to the red target point."]
        # elif skill_phase >= 2:
        #     options = [
        #         "a robot arm supporting the lever.",
        #     ]

    elif task_name == "peg-insert-side":
        if skill_phase == 0:
            options = ["a robot arm grabbing the green peg."]
        elif skill_phase == 1:
            options = ["a robot arm lifting the green peg from the floor."]
        elif skill_phase == 2:
            options = ["a robot arm inserting the green peg to the hole of the red box."]
        elif skill_phase >= 3:
            options = [
                "a robot arm holding the green peg after inserting.",
            ]

    elif task_name == "push":
        if skill_phase == 0:
            options = ["a robot arm grabbing the red cube."]
        elif skill_phase == 1:
            options = ["a robot arm pushing the grabbed red cube to the green target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the grabbed red cube near the green target point.",
            ]

    elif task_name == "push-back":
        if skill_phase == 0:
            options = ["a robot arm grabbing the red cube."]
        elif skill_phase == 1:
            options = ["a robot arm pushing the grabbed red cube to the green target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the grabbed red cube near the green target point.",
            ]

    elif task_name == "faucet-close":
        if skill_phase == 0:
            options = ["a robot arm reaching the faucet handle."]
        elif skill_phase >= 1:
            options = ["a robot arm rotating the faucet handle to the right."]
        # elif skill_phase >= 2:
        #     options = [
        #         "a robot arm holding the faucet handle after rotating.",
        # ]

    elif task_name == "disassemble":
        if skill_phase == 0:
            options = ["a robot arm grabbing the round peg."]
        elif skill_phase == 1:
            options = ["a robot arm extracting the round peg from the round nut."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the round peg above the round nut.",
            ]

    elif task_name == "sweep-into":
        if skill_phase == 0:
            options = ["a robot arm grabbing the red cube."]
        elif skill_phase == 1:
            options = ["a robot arm sweeping the grabbed red cube to the blue target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm holding the grabbed red cube near the blue target point.",
            ]

    elif task_name == "hand-insert":
        if skill_phase == 0:
            options = ["a robot arm grabbing the red cube."]
        elif skill_phase == 1:
            options = ["a robot arm sweeping the grabbed red cube to the blue target point."]
        elif skill_phase >= 2:
            options = [
                "a robot arm inserting the red cube to the blue target point in the hole.",
            ]

    if skill_phase == 999:
        options = ["video of the robot arm failed to complete the task."] * 20
    assert options is not None, f"task_name {task_name} / skill_phase: {skill_phase}"
    if output_type == "one":
        return np.random.choice(options, 1).item()
    elif output_type == "all":
        return options


def get_rlbench_instruct(task_name, skill_phase, output_type="one"):
    options = None
    if task_name == "pick_up_cup":
        if skill_phase == 0:
            options = ["a robot arm grasping the red cup."]
        elif skill_phase == 1:
            options = ["a robot arm lifting the grasping red cup from the table."]
        elif skill_phase >= 2:
            options = ["a robot arm holding the red cup after lifting."]

    elif task_name == "phone_on_base":
        if skill_phase == 0:
            options = ["a robot arm grasping the handset."]
        elif skill_phase == 1:
            options = ["a robot arm lifting the grasped handset from the table."]
        elif skill_phase == 2:
            options = ["a robot arm moving the grasped handset towards the base."]
        elif skill_phase >= 3:
            options = ["a robot arm putting the handset on the base."]

    elif task_name == "take_umbrella_out_of_umbrella_stand":
        if skill_phase == 0:
            options = ["a robot arm grasping the umbrella."]
        elif skill_phase == 1:
            options = ["a robot arm taking the grasped umbrella ouf of the umbrella stand."]
        elif skill_phase >= 2:
            options = ["a robot arm holding the umbrella on the umbrella stand."]

    elif task_name == "put_rubbish_in_bin":
        if skill_phase == 0:
            options = ["a robot arm grasping the white rubbish."]
        elif skill_phase == 1:
            options = ["a robot arm lifting the grasped white rubbish from the table."]
        elif skill_phase >= 2:
            options = ["a robot arm putting the grasped white rubbish in the black bin."]

    elif task_name == "stack_wine":
        if skill_phase == 0:
            options = ["a robot arm grasping the wine bottle."]
        elif skill_phase == 1:
            options = ["a robot arm lifting the grasped wine bottle from the table."]
        elif skill_phase >= 2:
            options = ["a robot arm stacking the grasped wine bottle to the wine rack."]

    if skill_phase == 999:
        options = ["video of the robot arm failed to complete the task."] * 20
    assert options is not None, f"task_name {task_name} / skill_phase: {skill_phase}"
    if output_type == "one":
        return np.random.choice(options, 1).item()
    elif output_type == "all":
        return options


def get_maniskill_instruct(task_name, skill_phase, output_type="one"):
    options = None
    if task_name.startswith("PickSingle"):
        if skill_phase == 0:
            options = [
                "Grasp the target object using gripper of the robot arm.",
                "Hold the target item with the robot arm's gripper.",
                "Secure the object with the robot arm's gripper.",
                "Pick up the component using the robot's gripper.",
                "Clasp the item using the robotic gripper.",
                "Take hold of the piece with the robot arm.",
                "Grip the object using the robot arm's gripper.",
                "Seize the component with the gripper of the robot arm.",
                "Latch onto the item using the robotic gripper.",
                "Snatch the part with the robot arm's gripper.",
                "Capture the object using the robot's gripper.",
                "Acquire the item with the robotic gripper.",
                "Hold onto the component using the robot arm's gripper.",
                "Grapple the object using the robot's gripper.",
                "Clamp the item securely with the gripper.",
                "Take possession of the part with the robot arm.",
                "Embrace the object with the robotic gripper.",
                "Clench the component using the robot arm's gripper.",
                "Securely hold the item with the robot's gripper.",
                "Latch onto and lift the object with the gripper.",
            ]
        elif skill_phase == 1:
            options = [
                "Move the target object to the goal position using robot arm.",
                "Position the object at the goal with the robot arm.",
                "Place the component in the desired location with the robot arm.",
                "Transport the item to the designated spot using the robot arm.",
                "Relocate the object to the goal position with the robot arm.",
                "Transfer the component to the specified location using the robot arm.",
                "Shift the item to the target location using robot arm.",
                "Convey the object to the desired position with the robot arm.",
                "Place the part at the goal using the robotic arm.",
                "Adjust the component's position with the robot arm.",
                "Deliver the item to the specified spot using the robot arm.",
                "Reposition the object to the goal position with the robot arm.",
                "Transport and place the component using the robotic arm.",
                "Transfer the item to the intended location with the robot arm.",
                "Move the object to the goal area using robot arm.",
                "Relocate and position the component with the robotic arm.",
                "Shift and place the item at the goal using robot arm.",
                "Convey the object to the designated position using the robot arm.",
                "Accurately place the part at the goal with the robot arm.",
                "Precisely adjust the component's location using robot arm.",
            ]
        elif skill_phase >= 2:
            options = [
                "Keep the target object and the robot arm stationary.",
                "Maintain a stable position for the object and the robot arm.",
                "Ensure the target object and the robot arm remain still.",
                "Hold the object and the robot arm in a fixed position.",
                "Stabilize the target object and the robot arm.",
                "Keep both the object and the robot arm motionless.",
                "Secure the object and the robot arm in place.",
                "Prevent any movement of the target object and the robot arm.",
                "Lock the position of both the object and the robot arm.",
                "Anchor the target object and the robot arm in position.",
                "Freeze the object and the robot arm in their current locations.",
                "Sustain the stationary state of both the object and the robot arm.",
                "Immoblize the target object and the robot arm.",
                "Ensure that the object and the robot arm do not move.",
                "Place the object and the robot arm in a fixed state.",
                "Keep both the object and the robot arm at rest.",
                "Hold the position of the object and the robot arm steady.",
                "Maintain the static state of both the object and the robot arm.",
                "Steady both the object and the robot arm.",
                "Fix the object and the robot arm in place.",
            ]

    if task_name.startswith("TurnFaucet"):
        if skill_phase == 0:
            options = [
                "Move the target handle using robot arm.",
                "Adjust the handle with the robot arm.",
                "Shift the handle using the robot arm.",
                "Reposition the handle with the robot arm.",
                "Slide the handle with the robot arm.",
                "Transfer the handle using the robot arm.",
                "Relocate the handle with the robot arm.",
                "Shift the position of the handle with the robot arm.",
                "Displace the handle with the robot arm.",
                "Place the handle using the robot arm.",
                "Adjust the position of the handle with the robot arm.",
                "Shift the handle's location with the robot arm.",
                "Relocate the handle's position with the robot arm.",
                "Move the handle to a new position with the robot arm.",
                "Reposition the handle to the desired location using the robot arm.",
                "Slide the handle to the designated spot with the robot arm.",
                "Transfer the handle to the required position using the robot arm.",
                "Displace the handle to the desired location with the robot arm.",
                "Place the handle at the specified spot using the robot arm.",
                "Shift the handle to the intended position with the robot arm.",
            ]
        elif skill_phase >= 1:
            options = [
                "Rotate the target handle until reaching the target angle using robot arm.",
                "Turn the handle until reaching the desired angle with the robot arm.",
                "Twist the handle until reaching the target position using the robot arm.",
                "Spin the handle until reaching the specified angle with the robot arm.",
                "Revolve the handle until reaching the required angle using the robot arm.",
                "Rotate the handle until reaching the designated angle with the robot arm.",
                "Swivel the handle until reaching the target orientation using the robot arm.",
                "Rotate the knob until reaching the desired position with the robot arm.",
                "Manipulate the handle until reaching the target angle using the robot arm.",
                "Twirl the handle until reaching the desired orientation with the robot arm.",
                "Spin the knob until reaching the specified angle using the robot arm.",
                "Adjust the handle until reaching the target position with the robot arm.",
                "Operate the handle until reaching the required angle using the robot arm.",
                "Rotate the lever until reaching the designated position with the robot arm.",
                "Turn the crank until reaching the desired angle using the robot arm.",
                "Move the handle until reaching the specified orientation with the robot arm.",
                "Twist the knob until reaching the target angle using the robot arm.",
                "Rotate the switch until reaching the required position with the robot arm.",
                "Spin the dial until reaching the designated angle using the robot arm.",
                "Rotate the wheel until reaching the desired orientation with the robot arm.",
            ]

    assert options is not None, f"task_name {task_name} / skill_phase: {skill_phase}"
    if output_type == "one":
        return np.random.choice(options, 1).item()
    elif output_type == "all":
        return options


def get_furniturebench_instruct(task_name, skill_phase, output_type="one"):
    options = None
    if task_name == "one_leg":
        if skill_phase == 0:
            options = [
                "a robot arm picking up the white tabletop.",
                # "grasp the white tabletop and lift it off the surface with the robotic arm.",
                # "retrieve the white surface.",
                # "grab the white tabletop.",
                # "elevate the white surface.",
                # "hoist the white tabletop.",
                # "seize the white surface.",
                # "hold up the white tabletop.",
                # "snatch the white surface.",
                # "uplift the white tabletop.",
                # "raise the white surface.",
                # "with the robotic arm, seize and elevate the white tabletop.",
                # "use the robotic arm to grab and raise the white surface.",
                # "task the robotic arm to snatch and elevate the white tabletop.",
                # "have the robot arm grip and hoist the white surface.",
                # "with the robotic arm, clutch and elevate the white tabletop.",
                # "operate the robot arm to handle and raise the white surface.",
                # "direct the robot arm to take and elevate the white tabletop.",
                # "command the robot to grasp and uplift the white surface.",
                # "direct the robot to seize and hoist the white tabletop.",
            ]
        elif skill_phase == 1:
            options = [
                "a robot arm pushing the white tabletop to the front right corner.",
                # "move the white surface towards the front-right.",
                # "shift the white tabletop to the foremost right.",
                # "nudge the white surface forward and to the right.",
                # "position the white tabletop at the front-right corner.",
                # "place the white surface at the foremost right corner.",
                # "push forward and right the white tabletop.",
                # "reposition the white surface to the front right end.",
                # "guide the white tabletop to the nearest right corner.",
                # "realign the white surface to the right front corner.",
                # "shove the white tabletop towards the front-right.",
                # "adjust the white surface towards the right and front.",
                # "redirect the white tabletop to the front right region.",
                # "position the white surface at the right and forward corner.",
                # "situate the white tabletop at the leading right edge.",
                # "force the white surface towards the front right end.",
                # "change the white tabletop's position to the front right.",
                # "settle the white surface at the front right edge.",
                # "move the white tabletop to the corner ahead and right.",
                # "arrange the white surface at the front and right.",
            ]
        elif skill_phase == 2:
            options = [
                "a robot arm picking up the white leg.",
                # "grasp the white leg and lift it off the surface with the robotic arm.",
                # "retrieve the white leg.",
                # "grab the white leg.",
                # "elevate the white leg.",
                # "hoist the white leg.",
                # "seize the white leg.",
                # "hold up the white leg.",
                # "snatch the white leg.",
                # "uplift the white leg.",
                # "raise the white leg.",
                # "with the robotic arm, seize and elevate the white leg.",
                # "use the robotic arm to grab and raise the white leg.",
                # "task the robotic arm to snatch and elevate the white leg.",
                # "have the robot arm grip and hoist the white leg.",
                # "with the robotic arm, clutch and elevate the white leg.",
                # "operate the robot arm to handle and raise the white leg.",
                # "direct the robot arm to take and elevate the white leg.",
                # "command the robot to grasp and uplift the white leg.",
                # "direct the robot to seize and hoist the white leg.",
            ]
        elif skill_phase == 3:
            options = [
                "a robot arm inserting the white leg into screw hole.",
                # "place the white leg into the screw hole.",
                # "fit the white leg into the hole.",
                # "position the white leg in the screw opening.",
                # "slot the white leg into the hole.",
                # "slide the white leg into the screw socket.",
                # "set the white leg inside the screw cavity.",
                # "push the white leg into the hole.",
                # "slip the white leg into the screw aperture.",
                # "lodge the white leg into the hole.",
                # "steady the white leg into the screw recess.",
                # "align the white leg with the hole and insert.",
                # "put the white leg inside the screw hole.",
                # "stick the white leg into the screw pit.",
                # "tuck the white leg into the hole.",
                # "fix the white leg into the screw opening.",
                # "nestle the white leg into the screw niche.",
                # "sink the white leg into the hole.",
                # "press the white leg into the screw orifice.",
                # "nudge the white leg into the screw gap.",
            ]
        elif skill_phase == 4:
            options = [
                "a robot arm screwing the white leg until tightly fitted.",
                # "tighten the white leg until secure.",
                # "fasten the white leg until firmly joined.",
                # "turn the white leg until properly attached.",
                # "secure the white leg until well-settled.",
                # "twist the white leg until solidly connected.",
                # "bolt the white leg until tightly bonded.",
                # "rotate the white leg until completely assembled.",
                # "join the white leg until firmly in place.",
                # "lock the white leg until fully integrated.",
                # "affix the white leg until perfectly attached.",
                # "bind the white leg until entirely assembled.",
                # "latch the white leg until it's securely fitted.",
                # "stabilize the white leg until well-mounted.",
                # "combine the white leg until strongly joined.",
                # "connect the white leg until well-linked.",
                # "cinch the white leg until firmly bonded.",
                # "assemble the white leg until it's tightly fit.",
                # "integrate the white leg until closely attached.",
                # "mount the white leg until it's closely set.",
            ]
        elif skill_phase >= 5:
            options = [
                "a robot arm holding the white leg in place.",
            ]

    if task_name == "cabinet":
        if skill_phase == 0:
            options = [
                "grasp the cabinet box.",
                "seize the cabinet box.",
                "grip the cabinet box with the robotic arm.",
                "clutch the cabinet box.",
                "grab the cabinet box.",
                "hold the cabinet box.",
                "clasp the cabinet box with the robotic arm.",
                "snatch the cabinet box.",
                "latch onto the cabinet box.",
                "take hold of the cabinet box with the robotic arm.",
                "catch the cabinet box.",
                "secure the cabinet box.",
                "clamp onto the cabinet box with the robotic arm.",
                "embrace the cabinet box.",
                "hook onto the cabinet box.",
                "get a grip on the cabinet box.",
                "snag the cabinet box with the robotic arm.",
                "nab the cabinet box.",
                "envelop the cabinet box.",
                "clamp down on the cabinet box.",
            ]
        elif skill_phase == 1:
            options = [
                "place the cabinet box to the front right corner.",
                "set the cabinet box in the front right corner.",
                "position the cabinet box towards the front right.",
                "move the cabinet box to the forward right corner.",
                "locate the cabinet box at the front right side.",
                "put the cabinet box into the front right angle.",
                "shift the cabinet box to the right front corner.",
                "transfer the cabinet box to the front right spot.",
                "guide the cabinet box to the near right corner.",
                "slide the cabinet box to the front's right edge.",
                "transport the cabinet box to the right forward corner.",
                "direct the cabinet box to the front right region.",
                "arrange the cabinet box in the front right corner.",
                "deliver the cabinet box to the front right position.",
                "dispatch the cabinet box to the forward right side.",
                "relocate the cabinet box to the front right vicinity.",
                "nudge the cabinet box toward the front right corner.",
                "steer the cabinet box to the right-hand front corner.",
                "align the cabinet box with the front right corner.",
                "rest the cabinet box in the front right nook.",
            ]
        elif skill_phase == 2:
            options = [
                "pick up the left door.",
                "lift the left door.",
                "raise the left door with the robotic arm.",
                "grasp the left door and lift it off the surface with the robotic arm.",
                "seize the left door.",
                "clutch the left door.",
                "hoist the left door.",
                "hold the left door with the robotic arm.",
                "catch the left door.",
                "retrieve the left door.",
                "snatch the left door.",
                "with the robotic arm, seize and elevate the left door.",
                "with the robotic arm, take hold of the left door.",
                "grip the left door.",
                "secure the left door and lift it off.",
                "elevate the left door with the robotic arm.",
                "collect the left door.",
                "clasp the left door.",
                "haul up the left door.",
                "get a grip on the left door and lift it off the surface.",
            ]
        elif skill_phase == 3:
            options = [
                "insert the left door into the cabinet box.",
                "place the left door into the cabinet box.",
                "fit the left door into the cabinet box.",
                "slide the left door into the cabinet box.",
                "mount the left door on the cabinet box.",
                "set the left door into the cabinet box.",
                "position the left door in the cabinet box.",
                "put the left door into the cabinet box.",
                "affix the left door to the cabinet box.",
                "lodge the left door into the cabinet box.",
                "secure the left door on the cabinet box.",
                "fix the left door into the cabinet box.",
                "attach the left door to the cabinet box.",
                "embed the left door into the cabinet box.",
                "install the left door into the cabinet box.",
                "fasten the left door to the cabinet box.",
                "join the left door with the cabinet box.",
                "align the left door into the cabinet box.",
                "adjoin the left door to the cabinet box.",
                "enplace the left door into the cabinet box.",
            ]
        elif skill_phase == 4:
            options = [
                "pick up the right door.",
                "lift the right door.",
                "raise the right door with the robotic arm.",
                "grasp the right door and lift it off the surface with the robotic arm.",
                "seize the right door.",
                "clutch the right door.",
                "hoist the right door.",
                "hold the right door with the robotic arm.",
                "catch the right door.",
                "retrieve the right door.",
                "snatch the right door.",
                "with the robotic arm, seize and elevate the right door.",
                "with the robotic arm, take hold of the right door.",
                "grip the right door.",
                "secure the right door and lift it off.",
                "elevate the right door with the robotic arm.",
                "collect the right door.",
                "clasp the right door.",
                "haul up the right door.",
                "get a grip on the right door and lift it off the surface.",
            ]
        elif skill_phase == 5:
            options = [
                "insert the right door into the cabinet box.",
                "place the right door into the cabinet box.",
                "fit the right door into the cabinet box.",
                "slide the right door into the cabinet box.",
                "mount the right door on the cabinet box.",
                "set the right door into the cabinet box.",
                "position the right door in the cabinet box.",
                "put the right door into the cabinet box.",
                "affix the right door to the cabinet box.",
                "lodge the right door into the cabinet box.",
                "secure the right door on the cabinet box.",
                "fix the right door into the cabinet box.",
                "attach the right door to the cabinet box.",
                "embed the right door into the cabinet box.",
                "install the right door into the cabinet box.",
                "fasten the right door to the cabinet box.",
                "join the right door with the cabinet box.",
                "align the right door into the cabinet box.",
                "adjoin the right door to the cabinet box.",
                "enplace the right door into the cabinet box.",
            ]
        elif skill_phase == 6:
            options = [
                "make box stand upright",
                "set box in an upright position",
                "place box vertically",
                "position box to be upright",
                "erect the box",
                "stand the box on end",
                "put the box on its bottom",
                "make the box rise",
                "raise the box vertically",
                "prop the box upright",
                "upend the box",
                "place the box on its feet",
                "turn the box upright",
                "stand the box on its edge",
                "put the box on its end",
                "set the box to be vertical",
                "position the box in a standing orientation",
                "make the box go vertical",
                "adjust the box to stand",
                "convert the box to an upright position",
            ]
        elif skill_phase == 7:
            options = [
                "lift cabinet top using the robot arm",
                "grip the cabinet top with the robotic arm",
                "grasp the cabinet top.",
                "hoist the cabinet top.",
                "raise the cabinet top using the robot arm",
                "take the cabinet top with the robot arm",
                "elevate the cabinet top.",
                "secure the cabinet top with the robot arm",
                "hold the cabinet top with the robotic arm",
                "lift the cabinet lid.",
                "pick up the cabinet cover using the robotic arm",
                "raise the cabinet top with the robot arm",
                "retrieve the cabinet top",
                "collect the cabinet top with the robot arm",
                "seize the cabinet top using the robotic arm",
                "gather the cabinet top",
                "get the cabinet top with the robotic arm",
                "clasp the cabinet top",
                "take hold of the cabinet top",
                "acquire the cabinet top with the robot arm",
            ]
        elif skill_phase == 8:
            options = [
                "insert the cabinet top into screw hole.",
                "place the cabinet top into the screw hole.",
                "fit the cabinet top into the hole.",
                "position the cabinet top in the screw opening.",
                "slot the cabinet top into the hole.",
                "slide the cabinet top into the screw socket.",
                "set the cabinet top inside the screw cavity.",
                "push the cabinet top into the hole.",
                "slip the cabinet top into the screw aperture.",
                "lodge the cabinet top into the hole.",
                "steady the cabinet top into the screw recess.",
                "align the cabinet top with the hole and insert.",
                "put the cabinet top inside the screw hole.",
                "stick the cabinet top into the screw pit.",
                "tuck the cabinet top into the hole.",
                "fix the cabinet top into the screw opening.",
                "nestle the cabinet top into the screw niche.",
                "sink the cabinet top into the hole.",
                "press the cabinet top into the screw orifice.",
                "nudge the cabinet top into the screw gap.",
            ]
        elif skill_phase >= 9:
            options = [
                "screw the cabinet top.",
                "tighten the cabinet top until secure.",
                "fasten the cabinet top until firmly joined.",
                "turn the cabinet top until properly attached.",
                "secure the cabinet top until well-settled.",
                "twist the cabinet top until solidly connected.",
                "bolt the cabinet top until tightly bonded.",
                "rotate the cabinet top until completely assembled.",
                "join the cabinet top until firmly in place.",
                "lock the cabinet top until fully integrated.",
                "affix the cabinet top until perfectly attached.",
                "bind the cabinet top until entirely assembled.",
                "latch the cabinet top until it's securely fitted.",
                "stabilize the cabinet top until well-mounted.",
                "combine the cabinet top until strongly joined.",
                "connect the cabinet top until well-linked.",
                "cinch the cabinet top until firmly bonded.",
                "assemble the cabinet top until it's tightly fit.",
                "integrate the cabinet top until closely attached.",
                "mount the cabinet top until it's closely set.",
            ]

    if task_name == "desk":
        if skill_phase == 0:
            options = [
                "pick up the white tabletop.",
                "grasp the white tabletop and lift it off the surface with the robotic arm.",
                "retrieve the white surface.",
                "grab the white tabletop.",
                "elevate the white surface.",
                "hoist the white tabletop.",
                "seize the white surface.",
                "hold up the white tabletop.",
                "snatch the white surface.",
                "uplift the white tabletop.",
                "raise the white surface.",
                "with the robotic arm, seize and elevate the white tabletop.",
                "use the robotic arm to grab and raise the white surface.",
                "task the robotic arm to snatch and elevate the white tabletop.",
                "have the robot arm grip and hoist the white surface.",
                "with the robotic arm, clutch and elevate the white tabletop.",
                "operate the robot arm to handle and raise the white surface.",
                "direct the robot arm to take and elevate the white tabletop.",
                "command the robot to grasp and uplift the white surface.",
                "direct the robot to seize and hoist the white tabletop.",
            ]
        elif skill_phase == 1:
            options = [
                "push the white tabletop to the front right corner.",
                "move the white surface towards the front-right.",
                "shift the white tabletop to the foremost right.",
                "nudge the white surface forward and to the right.",
                "position the white tabletop at the front-right corner.",
                "place the white surface at the foremost right corner.",
                "push forward and right the white tabletop.",
                "reposition the white surface to the front right end.",
                "guide the white tabletop to the nearest right corner.",
                "realign the white surface to the right front corner.",
                "shove the white tabletop towards the front-right.",
                "adjust the white surface towards the right and front.",
                "redirect the white tabletop to the front right region.",
                "position the white surface at the right and forward corner.",
                "situate the white tabletop at the leading right edge.",
                "force the white surface towards the front right end.",
                "change the white tabletop's position to the front right.",
                "settle the white surface at the front right edge.",
                "move the white tabletop to the corner ahead and right.",
                "arrange the white surface at the front and right.",
            ]
        elif skill_phase in [2, 5, 9, 12]:
            options = [
                "pick up the white leg.",
                "grasp the white leg and lift it off the surface with the robotic arm.",
                "retrieve the white leg.",
                "grab the white leg.",
                "elevate the white leg.",
                "hoist the white leg.",
                "seize the white leg.",
                "hold up the white leg.",
                "snatch the white leg.",
                "uplift the white leg.",
                "raise the white leg.",
                "with the robotic arm, seize and elevate the white leg.",
                "use the robotic arm to grab and raise the white leg.",
                "task the robotic arm to snatch and elevate the white leg.",
                "have the robot arm grip and hoist the white leg.",
                "with the robotic arm, clutch and elevate the white leg.",
                "operate the robot arm to handle and raise the white leg.",
                "direct the robot arm to take and elevate the white leg.",
                "command the robot to grasp and uplift the white leg.",
                "direct the robot to seize and hoist the white leg.",
            ]
        elif skill_phase in [3, 6, 10, 13]:
            options = [
                "insert the white leg into screw hole.",
                "place the white leg into the screw hole.",
                "fit the white leg into the hole.",
                "position the white leg in the screw opening.",
                "slot the white leg into the hole.",
                "slide the white leg into the screw socket.",
                "set the white leg inside the screw cavity.",
                "push the white leg into the hole.",
                "slip the white leg into the screw aperture.",
                "lodge the white leg into the hole.",
                "steady the white leg into the screw recess.",
                "align the white leg with the hole and insert.",
                "put the white leg inside the screw hole.",
                "stick the white leg into the screw pit.",
                "tuck the white leg into the hole.",
                "fix the white leg into the screw opening.",
                "nestle the white leg into the screw niche.",
                "sink the white leg into the hole.",
                "press the white leg into the screw orifice.",
                "nudge the white leg into the screw gap.",
            ]
        elif skill_phase in [4, 7, 11, 14] or skill_phase > 14:
            options = [
                "screw the white leg.",
                "tighten the white leg until secure.",
                "fasten the white leg until firmly joined.",
                "turn the white leg until properly attached.",
                "secure the white leg until well-settled.",
                "twist the white leg until solidly connected.",
                "bolt the white leg until tightly bonded.",
                "rotate the white leg until completely assembled.",
                "join the white leg until firmly in place.",
                "lock the white leg until fully integrated.",
                "affix the white leg until perfectly attached.",
                "bind the white leg until entirely assembled.",
                "latch the white leg until it's securely fitted.",
                "stabilize the white leg until well-mounted.",
                "combine the white leg until strongly joined.",
                "connect the white leg until well-linked.",
                "cinch the white leg until firmly bonded.",
                "assemble the white leg until it's tightly fit.",
                "integrate the white leg until closely attached.",
                "mount the white leg until it's closely set.",
            ]
        elif skill_phase == 8:
            options = [
                "Turn the white tabletop 180 degrees to the right.",
                "Swivel the white tabletop half a turn clockwise.",
                "Spin the white tabletop 180 degrees rightward.",
                "Twist the white tabletop a half revolution clockwise.",
                "Pivot the white tabletop 180 degrees in a clockwise direction.",
                "Revolve the white tabletop 180 degrees right.",
                "Wheel the white tabletop a half circle clockwise.",
                "Rotate the white tabletop a semi-circle to the right.",
                "Circle the white tabletop 180 degrees clockwise.",
                "Swing the white tabletop half-round to the right.",
                "Roll the white tabletop 180 degrees in the clock's direction.",
                "Whirl the white tabletop half a rotation clockwise.",
                "Turn over the white tabletop clockwise.",
                "Flip the white tabletop 180 degrees right.",
                "Screw the white tabletop a half turn clockwise.",
                "Twirl the white tabletop 180 degrees clockwise.",
                "Invert the white tabletop halfway to the right.",
                "Gyrate the white tabletop 180 degrees clockwise.",
                "Rotate the white tabletop right for half a turn.",
                "Swish the white tabletop 180 degrees in a clock hand motion.",
            ]

    if task_name == "lamp":
        if skill_phase == 0:
            options = [
                "pick up the white base.",
                "grasp the white base and lift it off the surface with the robotic arm.",
                "retrieve the white surface.",
                "grab the white base.",
                "elevate the white surface.",
                "hoist the white base.",
                "seize the white surface.",
                "hold up the white base.",
                "snatch the white surface.",
                "uplift the white base.",
                "raise the white surface.",
                "with the robotic arm, seize and elevate the white base.",
                "use the robotic arm to grab and raise the white surface.",
                "task the robotic arm to snatch and elevate the white base.",
                "have the robot arm grip and hoist the white surface.",
                "with the robotic arm, clutch and elevate the white base.",
                "operate the robot arm to handle and raise the white surface.",
                "direct the robot arm to take and elevate the white base.",
                "command the robot to grasp and uplift the white surface.",
                "direct the robot to seize and hoist the white base.",
            ]
        elif skill_phase == 1:
            options = [
                "push the white base to the front right corner.",
                "move the white surface towards the front-right.",
                "shift the white base to the foremost right.",
                "nudge the white surface forward and to the right.",
                "position the white base at the front-right corner.",
                "place the white surface at the foremost right corner.",
                "push forward and right the white base.",
                "reposition the white surface to the front right end.",
                "guide the white base to the nearest right corner.",
                "realign the white surface to the right front corner.",
                "shove the white base towards the front-right.",
                "adjust the white surface towards the right and front.",
                "redirect the white base to the front right region.",
                "position the white surface at the right and forward corner.",
                "situate the white base at the leading right edge.",
                "force the white surface towards the front right end.",
                "change the white base's position to the front right.",
                "settle the white surface at the front right edge.",
                "move the white base to the corner ahead and right.",
                "arrange the white surface at the front and right.",
            ]
        elif skill_phase == 2:
            options = [
                "pick up the bulb.",
                "grasp the bulb and lift it off the surface with the robotic arm.",
                "retrieve the bulb.",
                "grab the bulb.",
                "elevate the bulb.",
                "hoist the bulb.",
                "seize the bulb.",
                "hold up the bulb.",
                "snatch the bulb.",
                "uplift the bulb.",
                "raise the bulb.",
                "with the robotic arm, seize and elevate the bulb.",
                "use the robotic arm to grab and raise the bulb.",
                "task the robotic arm to snatch and elevate the bulb.",
                "have the robot arm grip and hoist the bulb.",
                "with the robotic arm, clutch and elevate the bulb.",
                "operate the robot arm to handle and raise the bulb.",
                "direct the robot arm to take and elevate the bulb.",
                "command the robot to grasp and uplift the bulb.",
                "direct the robot to seize and hoist the bulb.",
            ]
        elif skill_phase == 3:
            options = [
                "insert the bulb into screw hole.",
                "place the bulb into the screw hole.",
                "fit the bulb into the hole.",
                "position the bulb in the screw opening.",
                "slot the bulb into the hole.",
                "slide the bulb into the screw socket.",
                "set the bulb inside the screw cavity.",
                "push the bulb into the hole.",
                "slip the bulb into the screw aperture.",
                "lodge the bulb into the hole.",
                "steady the bulb into the screw recess.",
                "align the bulb with the hole and insert.",
                "put the bulb inside the screw hole.",
                "stick the bulb into the screw pit.",
                "tuck the bulb into the hole.",
                "fix the bulb into the screw opening.",
                "nestle the bulb into the screw niche.",
                "sink the bulb into the hole.",
                "press the bulb into the screw orifice.",
                "nudge the bulb into the screw gap.",
            ]
        elif skill_phase == 4:
            options = [
                "screw the bulb.",
                "tighten the bulb until secure.",
                "fasten the bulb until firmly joined.",
                "turn the bulb until properly attached.",
                "secure the bulb until well-settled.",
                "twist the bulb until solidly connected.",
                "bolt the bulb until tightly bonded.",
                "rotate the bulb until completely assembled.",
                "join the bulb until firmly in place.",
                "lock the bulb until fully integrated.",
                "affix the bulb until perfectly attached.",
                "bind the bulb until entirely assembled.",
                "latch the bulb until it's securely fitted.",
                "stabilize the bulb until well-mounted.",
                "combine the bulb until strongly joined.",
                "connect the bulb until well-linked.",
                "cinch the bulb until firmly bonded.",
                "assemble the bulb until it's tightly fit.",
                "integrate the bulb until closely attached.",
                "mount the bulb until it's closely set.",
            ]
        elif skill_phase == 5:
            options = [
                "pick up the hood.",
                "grasp the hood and lift it off the surface with the robotic arm.",
                "retrieve the hood.",
                "grab the hood.",
                "elevate the hood.",
                "hoist the hood.",
                "seize the hood.",
                "hold up the hood.",
                "snatch the hood.",
                "uplift the hood.",
                "raise the hood.",
                "with the robotic arm, seize and elevate the hood.",
                "use the robotic arm to grab and raise the hood.",
                "task the robotic arm to snatch and elevate the hood.",
                "have the robot arm grip and hoist the hood.",
                "with the robotic arm, clutch and elevate the hood.",
                "operate the robot arm to handle and raise the hood.",
                "direct the robot arm to take and elevate the hood.",
                "command the robot to grasp and uplift the hood.",
                "direct the robot to seize and hoist the hood.",
            ]
        elif skill_phase >= 6:
            options = [
                "place the hood on the top of the base.",
                "Set the hood atop the base.",
                "Position the hood over the base.",
                "Mount the hood on the base.",
                "Rest the hood on the base top.",
                "Put the hood onto the base.",
                "Lay the hood on the base.",
                "Fit the hood on the base's top.",
                "Arrange the hood over the base.",
                "Install the hood on top of the base.",
                "Affix the hood to the base.",
                "Align the hood with the base top.",
                "Situate the hood on the base.",
                "Deposit the hood atop the base.",
                "Place the hood onto the base's surface.",
                "Locate the hood on the base.",
                "Secure the hood on the base.",
                "Attach the hood to the base top.",
                "Gently set the hood on the base.",
                "Drop the hood onto the base.",
            ]

    if task_name == "round_table":
        if skill_phase == 0:
            options = [
                "pick up the white tabletop.",
                "grasp the white tabletop and lift it off the surface with the robotic arm.",
                "retrieve the white surface.",
                "grab the white tabletop.",
                "elevate the white surface.",
                "hoist the white tabletop.",
                "seize the white surface.",
                "hold up the white tabletop.",
                "snatch the white surface.",
                "uplift the white tabletop.",
                "raise the white surface.",
                "with the robotic arm, seize and elevate the white tabletop.",
                "use the robotic arm to grab and raise the white surface.",
                "task the robotic arm to snatch and elevate the white tabletop.",
                "have the robot arm grip and hoist the white surface.",
                "with the robotic arm, clutch and elevate the white tabletop.",
                "operate the robot arm to handle and raise the white surface.",
                "direct the robot arm to take and elevate the white tabletop.",
                "command the robot to grasp and uplift the white surface.",
                "direct the robot to seize and hoist the white tabletop.",
            ]
        elif skill_phase == 1:
            options = [
                "push the white tabletop to the front right corner.",
                "move the white surface towards the front-right.",
                "shift the white tabletop to the foremost right.",
                "nudge the white surface forward and to the right.",
                "position the white tabletop at the front-right corner.",
                "place the white surface at the foremost right corner.",
                "push forward and right the white tabletop.",
                "reposition the white surface to the front right end.",
                "guide the white tabletop to the nearest right corner.",
                "realign the white surface to the right front corner.",
                "shove the white tabletop towards the front-right.",
                "adjust the white surface towards the right and front.",
                "redirect the white tabletop to the front right region.",
                "position the white surface at the right and forward corner.",
                "situate the white tabletop at the leading right edge.",
                "force the white surface towards the front right end.",
                "change the white tabletop's position to the front right.",
                "settle the white surface at the front right edge.",
                "move the white tabletop to the corner ahead and right.",
                "arrange the white surface at the front and right.",
            ]
        elif skill_phase == 2:
            options = [
                "pick up the white round leg.",
                "grasp the white round leg and lift it off the surface with the robotic arm.",
                "retrieve the white round leg.",
                "grab the white round leg.",
                "elevate the white round leg.",
                "hoist the white round leg.",
                "seize the white round leg.",
                "hold up the white round leg.",
                "snatch the white round leg.",
                "uplift the white round leg.",
                "raise the white round leg.",
                "with the robotic arm, seize and elevate the white round leg.",
                "use the robotic arm to grab and raise the white round leg.",
                "task the robotic arm to snatch and elevate the white round leg.",
                "have the robot arm grip and hoist the white round leg.",
                "with the robotic arm, clutch and elevate the white round leg.",
                "operate the robot arm to handle and raise the white round leg.",
                "direct the robot arm to take and elevate the white round leg.",
                "command the robot to grasp and uplift the white round leg.",
                "direct the robot to seize and hoist the white round leg.",
            ]
        elif skill_phase == 3:
            options = [
                "insert the white round leg into screw hole.",
                "place the white round leg into the screw hole.",
                "fit the white round leg into the hole.",
                "position the white round leg in the screw opening.",
                "slot the white round leg into the hole.",
                "slide the white round leg into the screw socket.",
                "set the white round leg inside the screw cavity.",
                "push the white round leg into the hole.",
                "slip the white round leg into the screw aperture.",
                "lodge the white round leg into the hole.",
                "steady the white round leg into the screw recess.",
                "align the white round leg with the hole and insert.",
                "put the white round leg inside the screw hole.",
                "stick the white round leg into the screw pit.",
                "tuck the white round leg into the hole.",
                "fix the white round leg into the screw opening.",
                "nestle the white round leg into the screw niche.",
                "sink the white round leg into the hole.",
                "press the white round leg into the screw orifice.",
                "nudge the white round leg into the screw gap.",
            ]
        elif skill_phase == 4:
            options = [
                "screw the white round leg.",
                "tighten the white round leg until secure.",
                "fasten the white round leg until firmly joined.",
                "turn the white round leg until properly attached.",
                "secure the white round leg until well-settled.",
                "twist the white round leg until solidly connected.",
                "bolt the white round leg until tightly bonded.",
                "rotate the white round leg until completely assembled.",
                "join the white round leg until firmly in place.",
                "lock the white round leg until fully integrated.",
                "affix the white round leg until perfectly attached.",
                "bind the white round leg until entirely assembled.",
                "latch the white round leg until it's securely fitted.",
                "stabilize the white round leg until well-mounted.",
                "combine the white round leg until strongly joined.",
                "connect the white round leg until well-linked.",
                "cinch the white round leg until firmly bonded.",
                "assemble the white round leg until it's tightly fit.",
                "integrate the white round leg until closely attached.",
                "mount the white round leg until it's closely set.",
            ]
        elif skill_phase == 5:
            options = [
                "pick up the white base.",
                "grasp the white base and lift it off the surface with the robotic arm.",
                "retrieve the white surface.",
                "grab the white base.",
                "elevate the white surface.",
                "hoist the white base.",
                "seize the white surface.",
                "hold up the white base.",
                "snatch the white surface.",
                "uplift the white base.",
                "raise the white surface.",
                "with the robotic arm, seize and elevate the white base.",
                "use the robotic arm to grab and raise the white surface.",
                "task the robotic arm to snatch and elevate the white base.",
                "have the robot arm grip and hoist the white surface.",
                "with the robotic arm, clutch and elevate the white base.",
                "operate the robot arm to handle and raise the white surface.",
                "direct the robot arm to take and elevate the white base.",
                "command the robot to grasp and uplift the white surface.",
                "direct the robot to seize and hoist the white base.",
            ]
        elif skill_phase == 6:
            options = [
                "insert the base into screw hole.",
                "place the base into the screw hole.",
                "fit the base into the hole.",
                "position the base in the screw opening.",
                "slot the base into the hole.",
                "slide the base into the screw socket.",
                "set the base inside the screw cavity.",
                "push the base into the hole.",
                "slip the base into the screw aperture.",
                "lodge the base into the hole.",
                "steady the base into the screw recess.",
                "align the base with the hole and insert.",
                "put the base inside the screw hole.",
                "stick the base into the screw pit.",
                "tuck the base into the hole.",
                "fix the base into the screw opening.",
                "nestle the base into the screw niche.",
                "sink the base into the hole.",
                "press the base into the screw orifice.",
                "nudge the base into the screw gap.",
            ]
        elif skill_phase >= 7:
            options = [
                "screw the base.",
                "tighten the base until secure.",
                "fasten the base until firmly joined.",
                "turn the base until properly attached.",
                "secure the base until well-settled.",
                "twist the base until solidly connected.",
                "bolt the base until tightly bonded.",
                "rotate the base until completely assembled.",
                "join the base until firmly in place.",
                "lock the base until fully integrated.",
                "affix the base until perfectly attached.",
                "bind the base until entirely assembled.",
                "latch the base until it's securely fitted.",
                "stabilize the base until well-mounted.",
                "combine the base until strongly joined.",
                "connect the base until well-linked.",
                "cinch the base until firmly bonded.",
                "assemble the base until it's tightly fit.",
                "integrate the base until closely attached.",
                "mount the base until it's closely set.",
            ]
    if skill_phase == 999:
        options = ["video of the robot arm failed to complete the task."] * 20

    assert options is not None, f"task_name {task_name} / skill_phase: {skill_phase}"
    if output_type == "one":
        return np.random.choice(options, 1).item()
    elif output_type == "all":
        return options
