import color_checker
import pyautogui 

class MenuNavigator():
    # Pyautogui uses absolute coordinates, so add back the missing pixels from the cropped screenshot
    CROPPED_X_PIXELS = 935
    CROPPED_Y_PIXELS = 0

    BATTLE_BUTTON_COLOR = (255, 187, 0)
    BATTLE_BUTTON_COORDINATES = (280 + CROPPED_X_PIXELS, 1065)

    OK_BUTTON_COLOR = (78, 175, 255)
    OK_BUTTON_COORDINATES = (325 + CROPPED_X_PIXELS, 1275)
    ALT_OK_BUTTON_COORDINATES = (440, 1275)

    PLAY_AGAIN_BUTTON_COLOR = (255, 200, 75)
    PLAY_AGAIN_BUTTON_COORDINATES = (330 + CROPPED_X_PIXELS, 1240)

    def __init__(self):
        self.color_checker = color_checker.ColorChecker()
        self.pyautogui = pyautogui

    def is_battle_button_present(self, screenshot):
        pixel_color = screenshot.getpixel(self.BATTLE_BUTTON_COORDINATES)

        if self.color_checker.is_color_close(pixel_color, self.BATTLE_BUTTON_COLOR):
            return True

        return False
    
    def is_ok_button_present(self, screenshot):
        pixel_color = screenshot.getpixel(self.OK_BUTTON_COORDINATES)

        if self.color_checker.is_color_close(pixel_color, self.OK_BUTTON_COLOR):
            return True

        return False
    
    def is_alt_ok_button_present(self, screenshot):
        pixel_color = screenshot.getpixel(self.ALT_OK_BUTTON_COORDINATES)

        if self.color_checker.is_color_close(pixel_color, self.OK_BUTTON_COLOR):
            return True

        return False
    
    def is_play_again_button_present(self, screenshot):
        pixel_color = screenshot.getpixel(self.PLAY_AGAIN_BUTTON_COORDINATES)

        if self.color_checker.is_color_close(pixel_color, self.PLAY_AGAIN_BUTTON_COLOR):
            return True

        return False
    
    def click_battle_button(self):
        self.pyautogui.click(self.BATTLE_BUTTON_COORDINATES, duration=0.1)

    def click_ok_button(self):
        self.pyautogui.click(self.OK_BUTTON_COORDINATES, duration=0.1)

    def click_alt_ok_button(self):
        self.pyautogui.click(self.ALT_OK_BUTTON_COORDINATES, duration=0.1)

    def click_play_again_button(self):
        self.pyautogui.click(self.PLAY_AGAIN_BUTTON_COORDINATES, duration=0.1)