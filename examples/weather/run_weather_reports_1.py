
# python run_weather_reports_1.py
from wvpy.jtools import JTask


if __name__ == '__main__':
    for state_code in ["CA", "TX", "KS"]:
        task = JTask(
            sheet_name="weather_example_1.ipynb",
            sheet_vars={"state_code": state_code},
            output_suffix=f"_{state_code}",
        )
        task.render_as_html()
