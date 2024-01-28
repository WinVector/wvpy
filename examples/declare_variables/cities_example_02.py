

from wvpy.jtools import JTask, run_pool


if __name__ == '__main__':
    cities = [
        "Los Angeles",
        "San Francisco",
        "Brisbane",
    ]
    tasks = [
        JTask(
            sheet_name="declare_example.ipynb",
            sheet_vars={"city": city},
            output_suffix=f"_{city}",
        )
        for city in cities
    ]
    run_pool(tasks)
