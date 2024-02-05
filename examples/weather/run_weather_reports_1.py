
# python run_weather_reports_1.py
from wvpy.jtools import JTask, run_pool


if __name__ == '__main__':
    state_codes = [
        "CA",
        "TX",
        "KS",
    ]
    tasks = [
        JTask(
            sheet_name="weather_example_1.ipynb",
            sheet_vars={"state_code": state_code},
            output_suffix=f"_{state_code}",
        )
        for state_code in state_codes
    ]
    run_pool(tasks)
