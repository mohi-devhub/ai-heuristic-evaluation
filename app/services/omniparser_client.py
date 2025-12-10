import logging
import asyncio
from typing import List, Dict, Any, Optional
from PIL import Image
import io
import json
from ultralytics import YOLO
import torch
from transformers import AutoProcessor, AutoModelForCausalLM



TYPE_MAPPING = {
    "clickable_button": "button",
    "icon_button": "button",
    "submit_button": "button",
    "button": "button",
    "text_input": "input",
    "input": "input",
    "search_bar": "input",
    "heading": "heading",
    "header": "heading",
    "title": "heading",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading"
}

logger = logging.getLogger(__name__)

class UIElement:
    def __init__(
        self,
        element_type: str,
        text: str,
        bounds: Dict[str, int],
        interactive: bool = False,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.element_type = element_type
        self.text = text
        self.bounds = bounds
        self.interactive = interactive
        self.attributes = attributes or {}

    def to_dict(self):
        return {
            "element_type": self.element_type,
            "text": self.text,
            "bounds": self.bounds,
            "interactive": self.interactive,
            "attributes": self.attributes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            element_type=data["element_type"],
            text=data.get("text", ""),
            bounds=data["bounds"],
            interactive=data.get("interactive", False),
            attributes=data.get("attributes", {})
        )

class UIElementDetectionResult:
    def __init__(
        self,
        elements: List[UIElement],
        layout_hierarchy: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.elements = elements
        self.layout_hierarchy = layout_hierarchy
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "elements": [e.to_dict() for e in self.elements],
            "layout_hierarchy": self.layout_hierarchy,
            "metadata": self.metadata
        }

class OmniParserClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_loaded = False

    async def initialize(self):
        self.logger.info("Initializing OmniParser client...")
        # Loading YOLO 
        self.yolo_model = YOLO("weights/icon_detect/best.pt")
        
        # Loading Florence-2 
        self.caption_model = AutoModelForCausalLM.from_pretrained(
            "weights/icon_caption_florence", 
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            "weights/icon_caption_florence", 
            trust_remote_code=True
        )
        self.model_loaded = True
        self.logger.info("OmniParser client initialized successfully")

    def _map_element_type(self, raw_type: str) -> str:
        return TYPE_MAPPING.get(raw_type.lower(), "unknown")

    async def detect_elements(
        self,
        image_data: bytes,
        image_url: Optional[str] = None
    ) -> UIElementDetectionResult:
        self.logger.info("Starting UI element detection...")

        if not self.model_loaded:
            await self.initialize()

        try:
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            self.logger.info(f"Processing image: {width}x{height}")

            results = self.yolo_model(image)
            elements = []
            for result in results:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get type 
                    cls_id = int(box.cls[0])
                    raw_type = self.yolo_model.names[cls_id]
                    mapped_type = self._map_element_type(raw_type)

                    # Create Element 
                    elements.append(UIElement(
                        element_type=mapped_type,
                        text="", 
                        bounds={"x": x1, "y": y1, "width": x2-x1, "height": y2-y1},
                        attributes={} 
                    ))

            layout_hierarchy = {}
            result = UIElementDetectionResult(
                elements=elements,
                layout_hierarchy=layout_hierarchy,
                metadata={
                    "width": width,
                    "height": height,
                    "total_elements": len(elements)
                }
            )

            self.logger.info(f"Detection complete: {len(elements)} elements found")
            return result

        except Exception as e:
            self.logger.error(f"Error in element detection: {str(e)}")
            raise

    def group_related_elements(self, elements: List[UIElement]) -> Dict[str, List[UIElement]]:
        grouped = {
            "buttons": [],
            "inputs": [],
            "navigation": [],
            "content": [],
            "links": []
        }

        for element in elements:
            element_type = element.element_type.lower()
            if element_type in ["button", "submit", "reset"]:
                grouped["buttons"].append(element)
            elif element_type in ["input", "textarea", "select"]:
                grouped["inputs"].append(element)
            elif element_type in ["nav", "menu", "header", "footer"]:
                grouped["navigation"].append(element)
            elif element_type in ["a", "link"]:
                grouped["links"].append(element)
            else:
                grouped["content"].append(element)

        return grouped
