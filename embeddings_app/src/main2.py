import os
from time import sleep
from dotenv import load_dotenv
import supervisely as sly
from supervisely.app.widgets import Progress, Button, Container, NotificationBox, RadioTable, Table

progress = Progress("Init message")
pbar = progress(message="2nd msg")

btn = Button("Start")
note = RadioTable(["A", "B"], [["a1", "b1"], ["a2", "b2"]])
note = Table({"A": ["a"]})
# note = Text("desc\r\n*desc*\n**sd**<br><br><b>B</b>")

content = Container([btn, progress, note])
app = sly.Application(layout=content)

# load_dotenv("local.env")
# load_dotenv(os.path.expanduser("~/supervisely.env"))

# api = sly.Api()

# print(api.file.list(440, "embeddings"))


# @btn.click
# def on_click():
#     pbar.update(1)
#     note.columns
#     pbar.set_postfix(ab=22)


@note.click
def clc(r):
    pass
