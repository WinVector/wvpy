
import os
import datetime
from multiprocessing import Pool
from wvpy.jtools import JTask, job_fn
import imgkit
import pdfkit


if __name__ == "__main__":
    # define the tasks
    tasks = [
        JTask(
            sheet_name="ParamExample",
            output_suffix=f"_{d}",
            exclude_input=True,
            init_code=f'import datetime\ndate = {d.__repr__()}',
        )
        for d in [datetime.date(2022, 8, 18), datetime.date(2022, 8, 19), datetime.date(2022, 8, 20)]
    ]
    # do the work
    with Pool(4) as p:
        p.map(job_fn, tasks)
    # convert to different formats
    for fname in os.listdir():
        if fname.startswith('ParamExample_') and fname.endswith('.html'):
            imgkit.from_file(fname, fname.removesuffix('.html') + ".png")
            pdfkit.from_file(fname, fname.removesuffix('.html') + ".pdf")
    
