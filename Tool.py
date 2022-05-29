from abc import abstractclassmethod, ABC
import math


class Tool(ABC):
    @abstractclassmethod
    def apply(self, *args, **kwargs):
        pass

    def normaliz_pixel(self,
    normalized_x,normalized_y,
    image_width,image_height):
        def is_valid_normalized_value(value: float):
            return (value > 0 or math.isclose(0, value)) and (
                value < 1 or math.isclose(1, value)
            )

        if not (
            is_valid_normalized_value(normalized_x)
            and is_valid_normalized_value(normalized_y)
        ):
            return None

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px


class FaceTool(Tool):
    @abstractclassmethod
    def apply(self, *args, **kwargs):
        pass


class EyeTool(FaceTool):
    @abstractclassmethod
    def apply(self, *args, **kwargs):
        pass
