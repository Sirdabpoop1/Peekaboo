from bing_image_downloader import downloader

categories = [
    "water bottles", "furniture", "trees", "shoes",
    "tools", "pets", "toys", "rocks", "hands", "clocks"
]

for category in categories:
    downloader.download(category, limit=100, output_dir='neg_data', adult_filter_off=True, force_replace=False, timeout=60)
