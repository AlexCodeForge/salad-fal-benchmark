"""Endpoints, prompts, and pricing mirrored from teselio-engine-rs analyze-fal constants.rs."""

SAM3_IMAGE_ENDPOINT = "fal-ai/sam-3/image"
EVF_SAM_ENDPOINT = "fal-ai/evf-sam"
DEPTH_ENDPOINT = "fal-ai/image-preprocessors/depth-anything/v2"
MIDAS_ENDPOINT = "fal-ai/image-preprocessors/midas"
MARIGOLD_DEPTH_ENDPOINT = "fal-ai/imageutils/marigold-depth"
VLM_ROOM_HEIGHT_ENDPOINT = "fal-ai/any-llm/vision"
VLM_ROOM_HEIGHT_MODEL = "google/gemini-2.5-flash"

WALL_PROMPT = "wall"
FLOOR_PROMPT = "floor"
WALL_SAM3_PROMPTS = ("wall", "molding", "mullion")
FLOOR_SAM3_PROMPTS = ("floor",)

MAX_WALL_MASKS = 8
MAX_FLOOR_MASKS = 8
MAX_WALL_SAM3_EXTRA_MASKS = 4
MAX_FLOOR_SAM3_EXTRA_MASKS = 4
MAX_OCCLUDER_MASKS = 8
MAX_WALLS = 8
MAX_FLOORS = 8

DEFAULT_MASK_SCORE = 0.5
MASK_THRESHOLD = 127
IOU_DEDUP_THRESH = 0.85
FRAGMENT_IOU_THRESH = 0.35
FRAGMENT_MIN_IOU = 0.05

WALL_HEIGHT_CLAMP_MIN_CM = 220.0
WALL_HEIGHT_CLAMP_MAX_CM = 400.0
WALL_HEIGHT_FALLBACK_CM = 260.0
MIN_VLM_CONFIDENCE = 0.4

WALL_HEIGHT_STUB_CM = 260.0
WALL_HEIGHT_STUB_SOURCE = "stub"

SAM3_MIN_SCORE = 0.25


def fal_price_usd(endpoint: str) -> float:
    if endpoint in (SAM3_IMAGE_ENDPOINT, EVF_SAM_ENDPOINT):
        return 0.005
    if endpoint == VLM_ROOM_HEIGHT_ENDPOINT:
        return 0.0025
    return 0.0
