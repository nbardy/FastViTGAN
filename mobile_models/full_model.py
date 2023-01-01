# Fast multi-modal GAN
import kornia
import random
#
# This is a model capable of training on multiple modalities
# We start with:
#  1. Masks
#  2. Images
#  3. CLIP Embeddings (Text and Image)
# Both are trained on the same MobileVit Encoder
# 4. Uses the MobileStyleGAN architecture as the decoder

# TODO: Add projected discriminator
# (https://github.com/autonomousvision/projected_gan)
# 5. Uses discriminator from ProjectedGAN
# discriminator
import math
from torch.utils.data import Dataset
import os
import random
from pg_modules.discriminator import ProjectedDiscriminator

# generator
from PIL import Image
import open_clip
import torch
from torch import nn
from decoder import MobileSynthesisNetwork
from mobilevit import MobileViTModel

mobile = MobileSynthesisNetwork()


# It uses a cross attention mechanism to fuse the modalities
#
# For Images it uses a MobileViT architecture to embed to the MobileViT space
# For Masks a single channel MobileViT architecture to embed to the MobileViT space
# For CLIP it learns a small linear model to map the embeddings to the MobileViT space
# The output is flattened then fed to the MobileStyleGAN decoder
# The it runs a cross attention mechanism to fuse the two modalities
#
# For CLIP it learns a small linear model to map the embeddings to the MobileViT space

# Dataset

# Hypers
default_hyperparameters = {
    "default_block_w_size": 3
}
SD_BATCH_SIZE = 4

# DatasetLoader in pytorch


class DatasetLoader(nn.torch):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __getitem__(self, index):
        return self.dataset[index]


def load_model():
    # Currently Biggest open clip
    # model_name = "ViT-H/14"
    model_name = 'ViT-B-32-quickgelu'

    # Load the model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='laion400m_e32')

    # Returns ClipModel
    return {
        "model": model,
        "preprocess": preprocess,
        "model_name": model_name,
        "model_config": open_clip.get_model_config(model_name)
    }


# clip_model = load_model()
def get_clip_image_features(clip_model):
    model_name = clip_model["model_name"]
    model = clip_model["model"]
    preprocess = clip_model["preprocess"]

    tokenizer = open_clip.get_tokenizer(model_name)

    image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        return image_features


def get_clip_image_features(clip_model, text):
    model_name = clip_model["model_name"]
    model = clip_model["model"]
    tokenizer = open_clip.get_tokenizer(model_name)

    text_tokens = tokenizer(text)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens)
        return text_features


def get_clip_image_features(clip_model, image_np):
    model = clip_model["model"]
    preprocess = clip_model["preprocess"]
    image = preprocess(image_np).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_image(image)
        return text_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rand_perlin_2d(shape, res, fade=lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(
        0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1, slice2): return gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(
        d[0], 0).repeat_interleave(d[1], 1)

    def dot(grad, shift): return (torch.stack((grid[:shape[0], :shape[1], 0] + shift[0],
                                               grid[:shape[0], :shape[1], 1] + shift[1]), dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


# Custom Dataset Loader
#
# For Training our model on the three modalities

# split, but with yield
def split_into_tiles(image, N):
    width, height = image.size
    for i in range(0, width, N):
        for j in range(0, height, N):
            box = (i, j, i+N, j+N)
            a = image.crop(box)
            yield a


def get_multiscale_tile_batch(image):
    # Take 2 tiles from the image
    # Take the first 2 tiles from a 2x zoom of the image.
    # Then take the next 2 tiles from a 4x zoom of the image
    image_tiles_0 = list(split_into_tiles(image, 256))
    image_tiles_0 = image_tiles_0[:4]

    image_tiles_1 = list(split_into_tiles(image.resize(
        (image.size[0] * 2, image.size[1] * 2)), 256))
    image_tiles_1 = image_tiles_1[:4]

    image_tiles_2 = list(split_into_tiles(image.resize(
        (image.size[0] * 4, image.size[1] * 4)), 256))
    image_tiles_2 = image_tiles_2[:4]

    return image_tiles_0 + image_tiles_1 + image_tiles_2


def get_multiscale_tile_batch(image_file_names):
    noise = rand_perlin_2d((256, 256), (8, 8))


def generate_sample(config):
    return {}

# Accepts a short text string of the style "rough; smooth", with up to 1-3 semicolons
#
# This function should take this text and modify it into a prompt that can be used
# to create beautiful textures with a stable diffusion model


def make_perfect_texture_prompt(text):
    # A rich prefix string for describing deta
    postfixes = ["realistic hyperdetailed 8 k hdr",
                 "seamless 4K texture", "rich, detailed texture"]

    postfix = random.choice(postfixes)

    return text + " " + postfix


# Uses stable diffusion to generate image with a given init and mask, and clip features
def generate_sd_image(init, mask, text):
    import diffusers

    pipeline = diffusers.StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable_diffusion_inpainting_256").to(device)

    prompt = make_perfect_texture_prompt(text)

    images = pipeline(init_image=init, mask=mask, prompt=prompt)

    for image in images:
        # import uuid4
        import uuid
        uuid = str(uuid.uuid4(()))

        # Save the image
        filename = f"{uuid}.png"
        image.save(filename)

        return filename

# smoothstep with thresholds


def smoothstep(x, low=0.0, high=1.0):
    x = (x - low) / (high - low)
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def generate_samples(config):
    base_tile = config["base_tile"]
    image_tile = config["image_tile"]

    sample_label = config["sample_label"]
    features = config["features"]
    mask = config["mask"]

    # For the inner mask use kornia filters to blur the
    # mask and then drop the lower 60% of the values to 0
    # with a smooth transition
    inner_mask = mask

    # Do a 2d blur fast with kornia
    inner_mask = kornia.filters.gaussian_blur2d(inner_mask, (3, 3))
    inner_mask = smoothstep(inner_mask, 0.6, 0.98)

    # Use the inner mask the paint the image_tile on the base_image
    # with the mask
    merged_image = base_tile * (1 - inner_mask) + image_tile * inner_mask

    # Set the redraw mask to be the initial mask subtracted by the inner mask
    redraw_mask = mask - inner_mask

    # Now use stable diffusion to redraw the image in the redraw mask
    merged_image_filenames = generate_sd_image(
        image=merged_image, mask=redraw_mask, clip_embedding=features, n=SD_BATCH_SIZE, prompt=sample_label)

    return merged_image_filenames


class MultiModalDatasetLoader(Dataset):
    def __init__(self, image_file_folder, image_input_size=256):
        self.image_size = 256
        clip_model = load_model()
        # Set dataset
        all_image_files = os.listdir(image_file_folder)
        self.dataset = []

        for image_file in all_image_files:
            # Get a 4 random base tiles
            base_image_file_names = random.sample(all_image_files, 4)
            # RGB
            image = Image.open(image_file).convert("RGB")
            label = image_file.split(".")[0]

            base_images = [Image.open(file) for file in base_image_file_names]

            image_tiles = get_multiscale_tile_batch(image)

            # Do for each base and concat
            base_tiles_nested = [get_multiscale_tile_batch(
                tile) for tile in base_images]
            # unroll nested list one layer
            base_tiles = [
                item for sublist in base_tiles_nested for item in sublist]

            # Get the CLIP embedding for the image and text
            text_features = get_clip_image_features(clip_model, label)
            image_features = get_clip_image_features(clip_model, image)

            # Split label encoded as ";" separated values into a list
            labels = label.split(";")

            # loop over base_tilesXimage_tiles
            for base_tile in base_tiles:
                # Save base tile as uuid
                uuid = str(uuid.uuid4(()))
                base_tile_filename = f"{uuid}.png"
                base_tile.save(base_tile_filename)

                for image_tile in image_tiles:
                    # Get n random labels for n from 0 to len(labels)
                    n = random.randint(0, len(labels))
                    random_labels = random.sample(labels, n)
                    sample_label = ";".join(random_labels)
                    # random text or image features

                    mask = rand_perlin_2d((256, 256), (8, 8))

                    # save as file
                    mask = Image.fromarray(mask)
                    uuid = str(uuid.uuid4(()))
                    mask_filename = f"{uuid}.png"
                    mask.save(mask_filename)

                    samples = generate_samples({
                        "base_tile": base_tile,
                        "image_tile": image_tile,
                        "text": sample_label,
                        "mask": mask,
                    })

                    for sample in samples():
                        uuid = str(uuid.uuid4(()))
                        sample_filename = f"{uuid}.png"
                        sample.save(sample_filename)

                        features = random.choice(
                            [text_features, image_features])
                        self.dataset.append({
                            "base_tile": base_tile_filename,
                            "taget_image_filename": sample_filename,
                            "mask_filename": mask_filename,
                            "clip_features": features,
                        })

    def __len__(self):
        return len(self.dataset)

    def _getitem__(self, index):
        self.mask = rand_perlin_2d((256, 256), (8, 8))

        datum = self.dataset[index]

        # For each item:
        # Loads an two images from the dataset
        # Generates a CLIP embedding for each image and it's text label


class MultiModalFastGAN(nn.torch):
    # init
    def __init__(self, image_size=(512, 512), mask_size=(512, 512), clip_embedding_size=None):
        if (clip_embedding_size is None):
            raise ValueError("clip_embedding_size must be specified")

        # MobileVIT output size is 8x8
        # Flatten to 64
        mobile_vit_output_size = 64

        # A fast linear model to map the clip embedding to the MobileVit space
        self.clip_linear_encoder = nn.Sequential(
            nn.Linear(clip_embedding_size, clip_embedding_size/2),
            nn.ReLU(),
            nn.Linear(clip_embedding_size/2, mobile_vit_output_size),
            nn.ReLU(),
            nn.Linear(clip_embedding_size, clip_embedding_size/2),
            nn.ReLU(),
        )
        # Linear(clip_embedding_size, mobile_vit_output_size)

        # MobileVit Encoder
        self.image_encoder = MobileViTModel()

        # Align
        # Model that concats and linear layers to align the three modalities
        self.align = nn.Linear(mobile_vit_output_size *
                               3, mobile_vit_output_size)

        # MobileStyleGAN Decoder
        self.decoder = MobileSynthesisNetwork(
            block_w_size=default_hyperparameters)

    def forward(self, x_image, x_mask, x_clip):

        z_image = self.image_encoder(x_image)
        # Repeat mask in 3 channels then encoder in image_encoder
        z_mask = self.image_encoder(x_mask.repeat(1, 3, 1, 1))
        # y_mask = y_mask[:, 0, :, :]

        # Linear model to map clip embedding to MobileVit space
        z_clip = self.linear_clip_model(x_clip)

        # Concatenate the three modalities
        z = torch.cat((z_image, z_mask, z_clip), dim=1)

        # Align the three modalities
        z = self.align(z)

        # Decode
        x = self.decoder(z)

        return x


def start():
    generator = MultiModalFastGAN(clip_embedding_size=512)

    D = ProjectedDiscriminator()
    D.feature_network.requires_grad_(False)
