from dotenv import load_dotenv
import os
from openai import OpenAI
import base64
import json
import time

# 加载上一级目录的 .env 文件
load_dotenv("../.env")

# 获取 Qwen (DashScope) 的 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    raise ValueError("❌ 未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")

# 图片目录路径
IMAGE_DIR = r"C:\Users\jinchi\PycharmProjects\WhereisWaldoAgent\outputs\eval_patches"

# 使用的 Qwen 视觉模型
MODEL_NAME = "qwen-vl-max"

# ============================================

def encode_image(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_qwen_vl(client, image_path):
    """调用 Qwen-VL API 检测 Waldo"""
    base64_image = encode_image(image_path)

    prompt = """
    请仔细观察这张图片。这是一张《威利在哪里？》(Where's Waldo?) 风格的插图。
    任务：判断图片中是否包含 Waldo。

    请严格按照以下 JSON 格式返回结果，不要包含其他多余内容：
    {
        "found": true 或 false,
    }
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content
        # 清理可能存在的 Markdown 代码块标记
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result_json = json.loads(content)
        return result_json, None

    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}\n模型原始返回: {content}"
    except Exception as e:
        return None, f"API 调用异常: {str(e)}"

def main():
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 获取所有正例图片
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_pos.jpg")]

    if not image_files:
        print(f"❌ 在目录 {IMAGE_DIR} 中未找到 _pos.jpg 文件。")
        return

    print(f"🔍 找到 {len(image_files)} 张正例图片，开始批量检测...\n")
    print("=" * 60)

    # 统计变量
    total = len(image_files)
    success_count = 0  # True Positive
    fail_count = 0     # False Negative
    error_count = 0

    # 计时：墙钟总耗时 + 纯 API 平均每张耗时
    t_wall_start = time.perf_counter()
    api_secs = 0.0

    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(IMAGE_DIR, img_name)
        print(f"\n📸 [{idx}/{total}] 正在检测: {img_name}")
        print("-" * 40)

        t0 = time.perf_counter()
        result, error = query_qwen_vl(client, img_path)
        api_secs += time.perf_counter() - t0

        if error:
            print(f"❌ 检测失败 (Error): {error}")
            error_count += 1
        else:
            found = result.get("found", False)
            bbox = result.get("bbox", "N/A")
            desc = result.get("description", "N/A")

            if found:
                # 成功找到 (True Positive)
                print(f"✅ 结果: 成功找到 Waldo! (True Positive)")
                print(f"📍 描述: {desc}")
                print(f"📦 坐标 (0-1000): {bbox}")
                success_count += 1
            else:
                # 未找到，即漏检 (False Negative)
                print(f"❌ 结果: 未找到 Waldo。 (False Negative / 漏检)")
                print(f"⚠️ 模型描述: {desc}")
                fail_count += 1

        print("=" * 60)

    # 打印最终统计汇总
    print("\n\n🎉 检测任务完成！统计汇总如下:")
    print("-" * 40)
    print(f"📊 总图片数 (Total Positives): {total}")
    print(f"✅ 成功识别 (True Positive): {success_count} (成功率/Recall: {success_count / total * 100:.2f}%)")
    print(f"❌ 识别失败 (False Negative): {fail_count} (漏检率: {fail_count / total * 100:.2f}%)")
    print(f"⚠️ API/解析错误: {error_count}")
    wall = time.perf_counter() - t_wall_start
    print(f"⏱️ 总耗时 (墙钟): {wall:.1f}s ({wall / 60:.1f}min)")
    print(f"⏱️ 纯 API 耗时: {api_secs:.1f}s | 平均 {api_secs / total:.2f}s/张")
    print("-" * 40)


if __name__ == "__main__":
    main()