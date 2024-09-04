import json

with open("/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_planning_kprcnn/000023.json") as f:
    data = json.load(f)

# Replace ".jpg" with ".rgb.jpg" if that's the correct extension 
# data["image_rgb"] = data["image_rgb"].replace(".rgb", "") + ".jpg" 

# Output the JSON in one line
with open("reformatted_output.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))  