
import os
import datetime
from multiprocessing import Pool
from wvpy.jtools import JTask, job_fn
import imgkit
import pdfkit
import pprint


if __name__ == "__main__":
    # define the tasks
    tasks = [
        JTask(
            sheet_name="ParamExample",
            output_suffix=f"_{value_map['d']}_{value_map['n']}",
            exclude_input=True,
            init_code=f"""
import datetime
worksheet_params = {repr(value_map)}
""", strict=False
        )
        for value_map in [
            {"n": 5, "d": datetime.date(2022, 8, 18)}, 
            {"n": 10, "d": datetime.date(2022, 8, 19)}, 
            {"n": 15, "d": datetime.date(2022, 8, 20)}, 
        ]
    ]
    print("starting tasks:")
    pprint.pprint(tasks)
    # do the work
    with Pool(4) as p:
        p.map(job_fn, tasks)
    # convert to different formats
    for fname in os.listdir():
        if fname.startswith('ParamExample_') and fname.endswith('.html'):
            imgkit.from_file(fname, fname.removesuffix('.html') + ".png")
            # pdfkit.from_file(fname, fname.removesuffix('.html') + ".pdf")
    
