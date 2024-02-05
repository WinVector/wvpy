
# Basic weather API example using [weather.gov](https://www.weather.gov/documentation/services-web-api).


"""end code"""

from wvpy.jtools import declare_task_variables


"""end code"""

# set some variables to default values we are willing to override
# we do this with the with context manager so that our Jupyter or IDE thinks these variables are defined in our environment
# this defines both the set of names we allow overriding of and default values so we can debug in and IDE
with declare_task_variables(globals()):
    # set what state we are querying for
    state_code = 'CA'


"""end code"""

# import our packages
import datetime
import requests
import pandas as pd
from IPython.display import display, Markdown
import seaborn as sns
import matplotlib.pyplot as plt


"""end code"""

# configure
html_render = False

def present(txt: str):
    print(txt)


"""end code"""

# do the query
response = requests.get(f"https://api.weather.gov/alerts/active?area={state_code}")
# get JSON response
json_data = response.json()
# convert to data frame
df = pd.json_normalize(json_data["features"])


"""end code"""

# get local query time
now = datetime.datetime.now()
tz = now.astimezone().tzinfo
time_stamp_str = now.strftime('%Y-%m-%d %H:%M:%S ') + str(tz)


"""end code"""

# format some result info
present(f"""
[weather.gov](https://www.weather.gov/documentation/services-web-api) alerts for {state_code} retrieved at {time_stamp_str}.
""")


"""end code"""

# show an excerpt of the returned data frame
display_cols = [
    "properties.parameters.NWSheadline",
    'properties.areaDesc',
    'properties.effective',
    'properties.severity',
    'properties.certainty',
    'properties.event',
]
if df.shape[0] > 0:
    display_df = df.loc[
        pd.isnull(df["properties.parameters.NWSheadline"]) == False,
        display_cols].reset_index(drop=True, inplace=False)
else:
    display_df = pd.DataFrame({col: [None] for col in display_cols})

present(display_df.loc[:, ["properties.parameters.NWSheadline"]].to_markdown())


"""end code"""

# plot
if html_render and (sum(pd.isnull(display_df["properties.severity"]) == False) > 0):
    ax = sns.histplot(data=display_df, x="properties.severity")
    plt.title(f"Weather severity distribution for {state_code}, retrieved at {time_stamp_str}")
    ax.set(xlabel='Severity')
    plt.show()


"""end code"""

# mark provenance
display_df['QUERY_STATE_CODE'] = state_code
display_df['QUERY_TIME_STAMP'] = time_stamp_str


"""end code"""

# save to CSV file (could also write to database)
display_df.to_csv(
    f"{state_code}_weather.csv",
    index=False,
)

