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
## Inference
for inference after edit your config file run pipeline.py code 
