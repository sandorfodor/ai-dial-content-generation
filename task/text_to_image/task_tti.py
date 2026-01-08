import asyncio
from datetime import datetime
from pathlib import Path

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - 'hd' creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    # 1. Create DIAL bucket client
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # Create output directory if it doesn't exist
        output_dir = Path(__file__).parent.parent.parent / "generated_images"
        output_dir.mkdir(exist_ok=True)
        
        # 2. Iterate through Images from attachments, download them and then save here
        for i, attachment in enumerate(attachments):
            if attachment.url:
                try:
                    print(f"📥 Downloading image {i+1}/{len(attachments)}: {attachment.title or 'untitled'}")
                    
                    # Download image bytes from DIAL bucket
                    image_bytes = await bucket_client.get_file(attachment.url)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_extension = "png"  # Default to PNG
                    
                    # Try to get extension from attachment type or title
                    if attachment.type:
                        if "jpeg" in attachment.type or "jpg" in attachment.type:
                            file_extension = "jpg"
                        elif "png" in attachment.type:
                            file_extension = "png"
                        elif "webp" in attachment.type:
                            file_extension = "webp"
                    
                    filename = f"generated_image_{timestamp}_{i+1}.{file_extension}"
                    file_path = output_dir / filename
                    
                    # Save image to local file
                    with open(file_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    # 3. Print confirmation that image has been saved locally
                    print(f"✅ Image saved: {file_path}")
                    print(f"   Size: {len(image_bytes)} bytes")
                    
                except Exception as e:
                    print(f"❌ Failed to download image {i+1}: {e}")
            else:
                print(f"⚠️  Skipping attachment {i+1}: No URL provided")


async def async_start() -> None:
    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="dall-e-3",  # DALL-E 3 for image generation
        api_key=API_KEY
    )
    
    # 2. Generate image for "Sunny day on Bali"
    print("🎨 Generating image: 'Sunny day on Bali'")
    
    message = Message(
        role=Role.USER,
        content="Sunny day on Bali"
    )
    
    try:
        response = client.get_completion([message])
        print("✅ Image generation completed!")
        print(f"Response: {response.content}")
        
        # 3. Get attachments from response and save generated images
        if response.custom_content and response.custom_content.attachments:
            print(f"\n📎 Found {len(response.custom_content.attachments)} generated image(s)")
            await _save_images(response.custom_content.attachments)
        else:
            print("⚠️  No attachments found in response")
            
    except Exception as e:
        print(f"❌ Error during image generation: {e}")

    print("\n" + "="*80)
    
    # 4. Try to configure the picture for output via `custom_fields` parameter
    print("🎨 Generating configured image with custom settings...")
    
    custom_fields = {
        "size": Size.width_rectangle,  # 1792x1024 - wide format
        "style": Style.vivid,          # More dramatic and hyper-real
        "quality": Quality.hd,         # High definition
        "n": 1                         # Number of images to generate
    }
    
    enhanced_message = Message(
        role=Role.USER,
        content="A breathtaking sunset over rice terraces in Bali, with traditional temples in the background, golden hour lighting, ultra-detailed, cinematic composition"
    )
    
    try:
        response = client.get_completion(
            messages=[enhanced_message],
            custom_fields=custom_fields
        )
        print("✅ Enhanced image generation completed!")
        print(f"Response: {response.content}")
        
        if response.custom_content and response.custom_content.attachments:
            print(f"\n📎 Found {len(response.custom_content.attachments)} enhanced image(s)")
            await _save_images(response.custom_content.attachments)
        else:
            print("⚠️  No attachments found in enhanced response")
            
    except Exception as e:
        print(f"❌ Error during enhanced image generation: {e}")

    print("\n" + "="*80)
    
    # 5. Test it with the 'imagegeneration@005' (Google image generation model)
    print("🎨 Testing with Google's imagegeneration@005 model...")
    
    google_client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="imagegeneration@005",
        api_key=API_KEY
    )
    
    google_message = Message(
        role=Role.USER,
        content="A serene beach scene in Bali with crystal clear turquoise water, white sand, palm trees swaying in the breeze, and a traditional Balinese boat in the distance"
    )
    
    # Google's image generation might have different custom fields
    google_custom_fields = {
        "aspectRatio": "16:9",  # Google uses different parameter names
        "seed": 42,             # For reproducible results
        "guidanceScale": 7.5    # Controls how closely the image follows the prompt
    }
    
    try:
        response = google_client.get_completion(
            messages=[google_message],
            custom_fields=google_custom_fields
        )
        print("✅ Google image generation completed!")
        print(f"Response: {response.content}")
        
        if response.custom_content and response.custom_content.attachments:
            print(f"\n📎 Found {len(response.custom_content.attachments)} Google-generated image(s)")
            await _save_images(response.custom_content.attachments)
        else:
            print("⚠️  No attachments found in Google response")
            
    except Exception as e:
        print(f"❌ Error during Google image generation: {e}")

    # Bonus: Generate multiple variations
    print("\n" + "="*80)
    print("🎨 Generating multiple style variations...")
    
    styles_to_test = [
        ("natural", "A peaceful morning in Bali with soft natural lighting"),
        ("vivid", "A dramatic and vibrant Bali landscape with intense colors")
    ]
    
    for style, prompt in styles_to_test:
        print(f"\n🎭 Generating {style} style image...")
        
        style_custom_fields = {
            "size": Size.square,
            "style": style,
            "quality": Quality.hd
        }
        
        style_message = Message(
            role=Role.USER,
            content=prompt
        )
        
        try:
            response = client.get_completion(
                messages=[style_message],
                custom_fields=style_custom_fields
            )
            
            if response.custom_content and response.custom_content.attachments:
                await _save_images(response.custom_content.attachments)
                print(f"✅ {style.capitalize()} style image saved!")
            
        except Exception as e:
            print(f"❌ Error generating {style} style: {e}")


def start() -> None:
    """Synchronous wrapper for the async functionality."""
    asyncio.run(async_start())


start()