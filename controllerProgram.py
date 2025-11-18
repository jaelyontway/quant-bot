import os
import shutil

while True:
    date = input('Enter a date (YYYY-MM-DD): ').strip()

    # Folder and filenames
    folder_name = f"NVDA_{date}"
    news_file = f"news_{date}.csv"
    price_file = f"nvda_prices_{date}.csv"
    
    # Full paths to source files
    news_src = os.path.join("data", "demo_data", folder_name, news_file)
    price_src = os.path.join("data", "demo_data", folder_name, price_file)

    # Check if both exist
    if os.path.isfile(news_src) and os.path.isfile(price_src):
        print("success")

        # Destinations (current working directory)
        news_dest = os.path.join(os.getcwd(), "feedcsv.csv")
        price_dest = os.path.join(os.getcwd(), "feedcsv2.csv")

        # Copy files
        shutil.copy(news_src, news_dest)
        shutil.copy(price_src, price_dest)

        print("Copied files successfully.")
        break

    else:
        print("demo does not exist — try again.")
