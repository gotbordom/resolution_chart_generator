import argparse
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image # Very important for legible fonts and text

# Base abstract class
class Tile:
    def __init__(self, size, line_thickness, line_color, background_color):
        self.width, self.height = size
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.background_color = background_color

    def draw(self):
        raise NotImplementedError("Subclasses must implement draw()")
    
# Concrete implementation of a Tile
# This one draws horizontal and vertical lines at a defined pixel width
class LinesTile(Tile):
    def __init__(self, size, line_thickness, line_color, background_color):
        super().__init__(size, line_thickness=line_thickness, line_color=line_color, 
                         background_color=background_color)

    def draw(self):
        img = np.ones((self.height, self.width), dtype=np.uint8) * self.background_color

        # Create pattern
        pattern = np.array([False] * self.line_thickness + [True] * self.line_thickness)

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
    def __init__(self, size, line_thickness, line_color, background_color):
        super().__init__(size, line_thickness=line_thickness, line_color=line_color, 
                         background_color=background_color)

    def draw(self):
        block_size = self.line_thickness
        cols = self.width // block_size
        rows = self.height // block_size

        # Calculate actual drawn area
        board_width = cols * block_size
        board_height = rows * block_size

        # Calculate offsets to center the pattern
        x_offset = (self.width - board_width) // 2
        y_offset = (self.height - board_height) // 2

        img = np.ones((self.height, self.width), dtype=np.uint8) * self.background_color

        for row in range(rows):
            for col in range(cols):
                if (row + col) % 2 == 0:
                    top = y_offset + row * block_size
                    left = x_offset + col * block_size
                    bottom = min(top + block_size, self.height)
                    right = min(left + block_size, self.width)
                    img[top:bottom, left:right] = self.line_color

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# THIS TILE DOESN'T USE LINE THICKENESS
# It always goes from center at 1 px to 24 px
class CircleLinesTile(Tile):
    def __init__(self, size, line_thickness, line_color, background_color,
                 num_lines = 8):
        super().__init__(size, line_thickness=line_thickness, line_color=line_color, 
                         background_color=background_color)
        self.num_lines = num_lines
        self.line_color = line_color
        self.background_color = background_color

    def draw(self):
        img = np.ones((self.height, self.width), dtype=np.uint8) * self.background_color

        whitespace = 15  # Pixels
        max_pixel = 24

        # Define outer radius
        radius = min(self.width, self.height) // 2 - whitespace
        inner_radius = radius // 8  # Start drawing from halfway out
        center = (self.width // 2, self.height // 2)

        for i in range(self.num_lines):
            angle = 2 * np.pi * i / self.num_lines
            dir_x = np.cos(angle)
            dir_y = np.sin(angle)

            for r in range(inner_radius, radius + 1):  # Include outer edge
                # Normalize r from 0 to 1 between inner and outer radius
                t = (r - inner_radius) / (radius - inner_radius)
                thickness = int(round(1 + (max_pixel - 1) * t))
                thickness = max(thickness, 1)

                x = int(round(center[0] + dir_x * r))
                y = int(round(center[1] + dir_y * r))

                if 0 <= x < self.width and 0 <= y < self.height:
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
        line_color = self.tile_kwargs.get("line_color", 0) # Default to black
        background_color = self.tile_kwargs.get("background_color", 255) # Default to white
        if tile_class is LinesTile:
            return tile_class(self.tile_size, self.line_thickness, 
                             line_color, background_color).draw()
        elif tile_class is ChessboardTile:
            return tile_class(self.tile_size, self.line_thickness, 
                             line_color, background_color).draw()
        elif tile_class is CircleLinesTile:
            return tile_class(self.tile_size, self.line_thickness, 
                             line_color, background_color).draw()
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
    parser = argparse.ArgumentParser(description="Generate a resolution chart for detecting blur.")
    parser.add_argument('--width', type=int, default=3880, help='Width of image to display')
    parser.add_argument('--height', type=int, default=880, help='Height of image to display')
    parser.add_argument('--line_thickness', type=int, default=24, help='Line thickness in pixels')
    parser.add_argument('--line_color', type=int, default=0, help='Line color in 0-255 scale')
    parser.add_argument('--background_color', type=int, default=255, help='Background color in 0-255 scale')

    args = parser.parse_args()

    # The layout of each rectangle within the resolution chart
    tiles = [
        [LinesTile, ChessboardTile], 
        [LinesTile, CircleLinesTile]
    ]

    chart = ResolutionChart(
        image_size=(args.width, args.height),
        rectangle_class=Rectangle,
        tile_classes = tiles,
        line_thickness=args.line_thickness,
        line_color=args.line_color,
        background_color=args.background_color,
        draw_border=True,
        border_thickness=2
    )
    chart_img = chart.draw()
    cv2.imshow("Resolution Chart", chart_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()