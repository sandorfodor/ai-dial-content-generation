import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


async def _put_image() -> Attachment:
    file_name = 'dialx-banner.png'
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = 'image/png'
    
    # 1. Create DialBucketClient
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # 2. Open image file
        with open(image_path, 'rb') as image_file:
            # 3. Use BytesIO to load bytes of image
            image_bytes = BytesIO(image_file.read())
            
            # 4. Upload file with client
            upload_result = await bucket_client.put_file(
                name=file_name,
                mime_type=mime_type_png,
                content=image_bytes
            )
            
            # 5. Return Attachment object with title (file name), url and type (mime type)
            file_url = upload_result.get('url')
            if not file_url:
                raise ValueError("No URL returned from file upload")
                
            return Attachment(
                title=file_name,
                url=file_url,
                type=mime_type_png
            )


async def async_start() -> None:
    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",  # Using GPT-4 with vision capabilities
        api_key=API_KEY
    )
    
    # 2. Upload image (use `_put_image` method)
    print("📤 Uploading image to DIAL bucket...")
    attachment = await _put_image()
    
    # 3. Print attachment to see result
    print("✅ Image uploaded successfully!")
    print(f"📎 Attachment details:")
    print(f"   Title: {attachment.title}")
    print(f"   URL: {attachment.url}")
    print(f"   Type: {attachment.type}")
    
    # 4. Call chat completion via client with list containing one Message
    print("\n🤖 Analyzing uploaded image...")
    
    message = Message(
        role=Role.USER,
        content="What do you see on this picture?",
        custom_content=CustomContent(attachments=[attachment])
    )
    
    try:
        response = client.get_completion([message])
        print("\n✅ Analysis result:")
        print(response.content)
    except Exception as e:
        print(f"❌ Error during image analysis: {e}")

    # Optional: Try with multiple models
    print("\n" + "="*80)
    print("🔄 Testing with different models...")
    
    models_to_test = [
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        "gemini-1.5-pro-002"
    ]
    
    for model in models_to_test:
        print(f"\n🧪 Testing with {model}...")
        try:
            test_client = DialModelClient(
                endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
                deployment_name=model,
                api_key=API_KEY
            )
            
            test_message = Message(
                role=Role.USER,
                content="Describe this image in one sentence.",
                custom_content=CustomContent(attachments=[attachment])
            )
            
            response = test_client.get_completion([test_message])
            print(f"✅ {model}: {response.content}")
            
        except Exception as e:
            print(f"❌ {model} failed: {e}")

    # Optional: Try with multiple pictures
    print("\n" + "="*80)
    print("🖼️  Testing with multiple attachments...")
    
    try:
        # Create a second attachment (reusing the same image for demo)
        attachment2 = await _put_image()
        
        multi_image_message = Message(
            role=Role.USER,
            content="Compare these two images. What similarities and differences do you notice?",
            custom_content=CustomContent(attachments=[attachment, attachment2])
        )
        
        response = client.get_completion([multi_image_message])
        print("✅ Multi-image analysis:")
        print(response.content)
        
    except Exception as e:
        print(f"❌ Multi-image analysis failed: {e}")


def start() -> None:
    """Synchronous wrapper for the async functionality."""
    asyncio.run(async_start())


start()