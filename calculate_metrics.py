import os

def parse_metrics_from_file(file_path):
    """
    從 metrics.txt 檔案中提取所有 8 項指標的數值。
    """
    metrics = {
        "FID": None,
        "LPIPS": None,
        "PSNR": None,
        "SSIM": None,
        "m-FID": None,
        "m-LPIPS": None,
        "m-PSNR": None,
        "m-SSIM": None,
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 提取指標數值
        for line in lines:
            for key in metrics.keys():
                if f"► {key}" in line:
                    metrics[key] = float(line.split(":")[1].strip())
    except FileNotFoundError:
        print(f"檔案 {file_path} 不存在！")
    except Exception as e:
        print(f"解析檔案 {file_path} 時發生錯誤: {e}")
    return metrics

def main():
    # 設定基礎路徑
    base_path = "metric_logs"
    all_metrics = {
        "FID": [],
        "LPIPS": [],
        "PSNR": [],
        "SSIM": [],
        "m-FID": [],
        "m-LPIPS": [],
        "m-PSNR": [],
        "m-SSIM": [],
    }

    # 遍歷 metric_logs 資料夾下的所有檔案
    for file_name in os.listdir(base_path):
        if file_name.startswith("renders_") and file_name.endswith("_metrics.txt"):  # 僅處理符合格式的檔案
            file_path = os.path.join(base_path, file_name)
            metrics = parse_metrics_from_file(file_path)
            if metrics:
                print(f"從 {file_path} 提取到指標: {metrics}")
                for key, value in metrics.items():
                    if value is not None:
                        all_metrics[key].append(value)

    # 計算每個指標的平均值
    print("\n=== 總體平均指標結果 ===")
    for key, values in all_metrics.items():
        if values:
            average = sum(values) / len(values)
            print(f"{key}: 平均值 = {average:.4f}")
        else:
            print(f"{key}: 無有效數值")

if __name__ == "__main__":
    main()