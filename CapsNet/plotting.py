import wandb
api = wandb.Api()

run = api.run("rhwas/object_detection/1pf98vz8")
if run.state == "finished":
   for k in run.history():
       print(k["_timestamp"], k["accuracy"])