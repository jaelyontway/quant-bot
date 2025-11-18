import os
import shutil
import time
import subprocess
import yaml  # for YAML modification

while True:
    date = input('Enter a date (YYYY-MM-DD): ').strip()

    folder_name = f"NVDA_{date}"
    news_file = f"news_{date}.csv"
    price_file = f"nvda_prices_{date}.csv"
    
    news_src = os.path.join("data", "demo_data", folder_name, news_file)
    price_src = os.path.join("data", "demo_data", folder_name, price_file)

    if os.path.isfile(news_src) and os.path.isfile(price_src):
        print("success")

        # Copy feeders into current working directory
        news_dest = os.path.join(os.getcwd(), "feedcsv.csv")
        price_dest = os.path.join(os.getcwd(), "feedcsv2.csv")

        shutil.copy(news_src, news_dest)
        shutil.copy(price_src, price_dest)
        print("Copied feeders into main directory.")
        time.sleep(0.5)

        # Run east model
        dest_folder = os.path.join(os.getcwd(), "src", "east")
        run_script = os.path.join(dest_folder, "run_east_model.py")
        if os.path.isfile(run_script):
            subprocess.run(["python3", run_script], cwd=dest_folder)
        else:
            print("run_east_model.py not found!")
            break

        # Simulation folder
        simulation_folder = os.path.join(os.getcwd(), "simulation")
        os.makedirs(simulation_folder, exist_ok=True)

        # Write date to targetSimDate.txt
        target_file = os.path.join(simulation_folder, "targetSimDate.txt")
        with open(target_file, "w") as f:
            f.write(date)

        # Copy config.yaml
        config_src = os.path.join(os.getcwd(), "config", "config.yaml")
        config_dest = os.path.join(simulation_folder, "config.yaml")
        nvidia_config_dest = os.path.join(simulation_folder, "nvidia-config.yaml")

        if os.path.isfile(config_src):
            shutil.copy(config_src, config_dest)
            shutil.copy(config_src, nvidia_config_dest)
            print("Config files copied to simulation folder.")
        else:
            print("config.yaml not found!")
            break

        # Update date in nvidia-config.yaml
        try:
            with open(nvidia_config_dest, 'r') as yf:
                config_data = yaml.safe_load(yf)

            config_data["date"] = date

            with open(nvidia_config_dest, 'w') as yf:
                yaml.dump(config_data, yf, default_flow_style=False)

            print("Updated simulation config date.")
        except Exception as e:
            print(f"Error updating config date: {e}")
            break

        # Copy price feeder into simulation folder and rename it
        sim_price_dest = os.path.join(simulation_folder, "NVDA_prices.csv")
        shutil.copy(price_dest, sim_price_dest)
        print("feedcsv2.csv copied into simulation as NVDA_prices.csv")
        time.sleep(0.5)
        # Run simulation_program.py
        sim_script = os.path.join(simulation_folder, "simulation_program.py")
        if os.path.isfile(sim_script):
            subprocess.run(["python3", sim_script], cwd=simulation_folder)
            print("Simulation complete.")
        else:
            print("simulation_program.py not found!")

        break

    else:
        print("demo does not exist — try again.")
