import base64
from pathlib import Path

from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",  # Using GPT-4 with vision capabilities
        api_key=API_KEY
    )

    # 2. Call client to analyze image with base64 encoded format
    print("🖼️  Analyzing local image (base64 encoded)...")
    
    base64_data_url = f"data:image/png;base64,{base64_image}"
    
    message_with_base64 = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What do you see on this picture?"),
            ImgContent(image_url=ImgUrl(url=base64_data_url))
        ]
    )
    
    try:
        response = client.get_completion([message_with_base64])
        print("\n✅ Response for base64 image:")
        print(response.content)
    except Exception as e:
        print(f"❌ Error with base64 image: {e}")

    print("\n" + "="*80 + "\n")

    # Try with URL: https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg
    print("🐘 Analyzing remote image (URL)...")
    
    elephant_url = "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg"
    
    message_with_url = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What do you see on this picture?"),
            ImgContent(image_url=ImgUrl(url=elephant_url))
        ]
    )
    
    try:
        response = client.get_completion([message_with_url])
        print("\n✅ Response for URL image:")
        print(response.content)
    except Exception as e:
        print(f"❌ Error with URL image: {e}")
