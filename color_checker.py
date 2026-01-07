class ColorChecker: 
    def __init__(self):
        pass

    def is_color_close(self, color, target, tolerance=80):
        for i in range(3):
            color_rgb_component = int(color[i])
            target_rgb_component = int(target[i])

            if abs(color_rgb_component - target_rgb_component) > tolerance:
                return False

        return True