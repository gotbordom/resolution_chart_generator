import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image # Very important for legible fonts and text

# Base abstract class
class Tile:
    def __init__(self, size, line_thickness, whitespace, line_color, background_color):
        self.width, self.height = size
        self.line_thickness = line_thickness
        self.whitespace = whitespace
        self.line_color = line_color
        self.background_color = background_color

    def draw(self):
        raise NotImplementedError("Subclasses must implement draw()")
    
# Concrete implementation of a Tile
# This one draws horizontal and vertical lines at a defined pixel width
class LinesTile(Tile):
    def __init__(self, size, line_thickness, whitespace, line_color, background_color):
        super().__init__(size, line_thickness=line_thickness, 
                         whitespace=whitespace, line_color=line_color, 
                         background_color=background_color)

    def draw(self):
        img = np.ones((self.height, self.width), dtype=np.uint8) * self.background_color

        # Create pattern
        pattern = np.array([False] * self.whitespace + [True] * self.line_thickness)

        # Horizontal lines (left half)
        h_repeats = (self.height + len(pattern) - 1) // len(pattern)
        h_mask = np.tile(pattern, h_repeats)[:self.height]
        img[h_mask, : self.width // 2] = self.line_color

        # Vertical lines (right half)
        v_repeats = (self.width - self.width // 2 + len(pattern) - 1) // len(pattern)
        v_pattern = np.concatenate([
            np.zeros(self.width // 2, dtype=bool),
            np.tile(pattern, v_repeats)[: self.width - self.width // 2]
        ])
        img[:, v_pattern] = self.line_color

        # Convert tile to color instead of grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

class ChessboardTile(Tile):
    def __init__(self, size, line_thickness, whitespace, line_color, background_color):
        super().__init__(size, line_thickness=line_thickness, 
                         whitespace=whitespace, line_color=line_color, 
                         background_color=background_color)

    def draw(self):
        block_size = self.line_thickness
        cols = self.width // block_size
        rows = self.height // block_size

        img = np.ones((self.height, self.width), dtype=np.uint8) * self.background_color

        for row in range(rows):
            for col in range(cols):
                if (row + col) % 2 == 0:
                    top = row * block_size
                    left = col * block_size
                    bottom = min((row + 1) * block_size, self.height)
                    right = min((col + 1) * block_size, self.width)
                    img[top:bottom, left:right] = self.line_color

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

class CircleLinesTile(Tile):
    def __init__(self, size, line_thickness, whitespace, line_color, background_color,
                 num_lines = 16):
        super().__init__(size, line_thickness=line_thickness, 
                         whitespace=whitespace, line_color=line_color, 
                         background_color=background_color)
        self.num_lines = num_lines
        self.line_color = line_color
        self.background_color = background_color

    # def draw(self):
    #     img = np.ones((self.height, self.width), dtype=np.uint8) * self.background_color
    #     center = (self.width // 2, self.height // 2)
    #     radius = min(self.width, self.height) // 2

    #     for i in range(self.num_lines):
    #         angle = 2 * np.pi * i / self.num_lines
    #         end_x = int(center[0] + radius * np.cos(angle))
    #         end_y = int(center[1] + radius * np.sin(angle))
    #         cv2.line(img, center, (end_x, end_y), self.line_color, thickness=1)

    #     return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def draw(self):
        img = np.ones((self.height, self.width), dtype=np.uint8) * self.background_color
        center = (self.width // 2, self.height // 2)
        radius = min(self.width, self.height) // 2

        for i in range(self.num_lines):
            angle = 2 * np.pi * i / self.num_lines
            # Direction vector of the ray
            dir_x = np.cos(angle)
            dir_y = np.sin(angle)

            # Draw points along the radius from center to edge
            for r in range(radius):
                # Interpolate thickness from 1 to self.line_thickness
                thickness = int(1 + (self.line_thickness/2 - 1) * (r / radius))
                if thickness < 1:
                    thickness = 1

                x = int(center[0] + dir_x * r)
                y = int(center[1] + dir_y * r)
                # Draw a small circle at this point to simulate growing line
                cv2.circle(img, (x, y), thickness // 2, self.line_color, -1)

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# Rectangles are then labeled and boardered sections of the resolution
# chart
class Rectangle:
    # A rectangle made of 2x2 tiles.
    def __init__(self, tile_classes, tile_size, label=None, label_position='center', line_thickness=1, draw_border=True, border_thickness=2, **tile_kwargs):
        self.label = label
        self.label_position = label_position
        self.line_thickness = line_thickness
        self.draw_border = draw_border
        self.border_thickness = border_thickness
        self.tile_kwargs = tile_kwargs
        self.tile_classes = tile_classes
        self.tile_size = tile_size

    def draw(self):
        # stack all tiles into the rectable
        opencv_color_red = (0,0,255)    # OpenCV uses (BGR)
        pil_color_red = (255,0,0)       # Pillow uses (RGB)
        row_images = []
        for row in self.tile_classes:
            row_tiles = [self.draw_tile(tile_class) for tile_class in row]
            row_images.append(np.hstack(row_tiles))
        rectangle_img = np.vstack(row_images)

        # draw label in center if provided
        if self.label:
            rectangle_img = self.draw_label(rectangle_img, "{} at {} px".format(self.label, self.line_thickness), 
                                            font_size=self.line_thickness, position=self.label_position, color=pil_color_red)


        # draw the border if enabled
        if self.draw_border:
            height, width, _ = rectangle_img.shape
            cv2.rectangle(
                rectangle_img,
                (0, 0),
                (width - 1, height - 1),
                opencv_color_red,
                self.border_thickness
            )
        
        return rectangle_img
    
    # This is the factory method for consturcting tiles.
    def draw_tile(self, tile_class):
        # Get shared kwargs
        whitespace = self.tile_kwargs.get("whitespace", 1)
        line_color = self.tile_kwargs.get("line_color", 0) # Default to black
        background_color = self.tile_kwargs.get("background_color", 255) # Default to white
        if tile_class is LinesTile:
            return tile_class(self.tile_size, self.line_thickness, 
                             whitespace, line_color, background_color).draw()
        elif tile_class is ChessboardTile:
            return tile_class(self.tile_size, self.line_thickness, 
                             whitespace, line_color, background_color).draw()
        elif tile_class is CircleLinesTile:
            return tile_class(self.tile_size, self.line_thickness, 
                             whitespace, line_color, background_color).draw()
        else:
            raise ValueError("Unknown tile class {}".format(tile_class))
                             

    def draw_label(self, img, text, font_size, position='center', color=(0, 0, 0)):
        # Convert OpenCV image (BGR) to PIL image (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Load a truetype font
        font_style = "DejaVuSans-Bold.ttf"
        try:
            font = ImageFont.truetype(font_style, font_size)
        except IOError:
            print("Error: Unable to load image font format: {}. Loading default instead.".format(font_style))
            font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        img_width, img_height = img_pil.size

        # Compute position
        if position == "top_left":
            origin = (5, 5)
        elif position == "top_right":
            origin = (img_width - text_width - 5, 5)
        elif position == "bottom_left":
            origin = (5, img_height - text_height - 5)
        elif position == "bottom_right":
            origin = (img_width - text_width - 5, img_height - text_height - 5)
        else:  # center
            origin = ((img_width - text_width) // 2, (img_height - text_height) // 2)


        # Draw the text
        draw.text(origin, text, font=font, fill=color)

        # Convert back to OpenCV image (BGR)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# The overall image that will get constructed and displayed
class ResolutionChart:
    # A chart made of 4x4 rectangles (each rectangle is 2x2 tiles).
    def __init__(self, image_size, rectangle_class, tile_classes, line_thickness, **tile_kwargs):
        self.image_size = image_size
        self.rectangle_class = rectangle_class
        self.tile_classes = tile_classes
        self.line_thickness = line_thickness
        self.tile_kwargs = tile_kwargs
        self.rows, self.cols = 4, 4
        # Column headers C, D, E, F
        self.col_labels = [chr(ord('C') + i) for i in range(self.cols)]
        # Row headers 1, 2, 3, 4
        self.row_labels = [str(i + 1) for i in range(self.rows)]

    def draw(self):
        rect_width = self.image_size[0] // self.cols
        rect_height = self.image_size[1] // self.rows
        tile_size = (rect_width // 2, rect_height // 2)

        # TODO (AT) While one lines are fancy, breaking this into two for loops 
        # in a clearer way is easier to read.
        grid = [
            [self.rectangle_class(
                tile_classes = self.tile_classes,
                tile_size = tile_size,
                label = "{0}{1}".format(self.col_labels[i], self.row_labels[j]),
                label_position = 'bottom_right',
                line_thickness = self.line_thickness,
                **self.tile_kwargs).draw() 
             for i in range(self.cols)]
            for j in range(self.rows)
        ]
        rows = [np.hstack(row) for row in grid]
        return np.vstack(rows)

def main():
    # The layout of each rectangle within the resolution chart
    tiles = [
        [LinesTile, ChessboardTile], 
        [LinesTile, CircleLinesTile]
    ]

    chart = ResolutionChart(
        image_size=(3880, 880),
        rectangle_class=Rectangle,
        tile_classes = tiles,
        line_thickness=24,
        whitespace=24,
        line_color=0,
        background_color=255,
        draw_border=True,
        border_thickness=2
    )
    chart_img = chart.draw()
    cv2.imshow("Resolution Chart", chart_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()