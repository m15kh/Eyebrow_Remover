## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

Edit the `config.json` file to set the input image path, output folder, line thickness, and line color:
```json
{
    "input_image_path": "test/tst1.png",
    "output_folder": "./output",
    "line_thickness": 1,
    "line_color": [255, 255, 255]
}
Here is the updated `README.md` file with the requested changes:

```markdown
## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

Edit the `config.json` file to set the input image path, output folder, line thickness, and line color:
```json
{
    "input_image_path": "test/tst1.png",
    "output_folder": "./output",
    "line_thickness": 1,
    "line_color": [255, 255, 255]
}
```

## Extracting Lines

If you want to extract the lines, uncomment the following two lines in `pipeline.py`:
```python
connect_noise_points(tim, p76, p50, (line_color), line_thickness)
connect_noise_points(tim, p82, p102, line_color, line_thickness)
```
<img src="img-readme/noise-coonect.jpg" alt="Noise Connect" width="512" height="512",> 