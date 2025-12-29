"""
End-to-End Prompt-to-ASCII Pipeline

Unified interface combining:
- Stable Diffusion image generation
- AISS and Random Forest structural mappers
- Multiple character sets
- Quality metrics

This is the main entry point for the library.
"""

from typing import Literal, Optional, Tuple, Union
from PIL import Image
import time

from .charsets import CharacterSet, get_charset, list_charsets
from .aiss import AISSMapper, create_aiss_mapper
from .random_forest import RandomForestMapper, create_random_forest_mapper
from .generator import PromptToImageGenerator, create_generator
from .preprocessing import resize_for_ascii, preprocess_for_structure
from .result import ASCIIResult, create_result
from .metrics import compute_ssim, character_diversity, compare_mappers


MapperType = Literal["aiss", "random_forest", "both"]


class PromptToASCII:
    """
    End-to-end Prompt-to-ASCII Art Generator.
    
    Combines Stable Diffusion image generation with structural
    character mapping to convert text prompts to ASCII art.
    
    Example:
        >>> pipeline = PromptToASCII(mapper="random_forest")
        >>> result = pipeline.generate("cyberpunk cityscape at night")
        >>> result.display()
        >>> result.save("output.html")
    """
    
    def __init__(
        self,
        mapper: MapperType = "random_forest",
        charset: str = "ascii_standard",
        tile_size: Tuple[int, int] = (8, 14),
        sd_model: str = "flux-schnell",  # FLUX.1 Schnell - best quality
        auto_train_rf: bool = True,
        rf_model_path: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            mapper: "aiss", "random_forest", or "both" for comparison
            charset: Character set name (see list_charsets())
            tile_size: Size of each character tile (width, height)
            sd_model: "flux-schnell" (best) or "sdxl-turbo" (faster)
            auto_train_rf: Automatically train RF if no model provided
            rf_model_path: Path to pre-trained RF model
        """
        self.mapper_type = mapper
        self.charset_name = charset
        self.tile_size = tile_size
        self.sd_model = sd_model
        
        # Load charset
        self._charset = get_charset(charset, tile_size)
        
        # Initialize mappers (lazy for AISS, eager for RF if auto_train)
        self._aiss_mapper: Optional[AISSMapper] = None
        self._rf_mapper: Optional[RandomForestMapper] = None
        
        if mapper in ("aiss", "both"):
            self._aiss_mapper = create_aiss_mapper(charset, tile_size)
        
        if mapper in ("random_forest", "both"):
            self._rf_mapper = create_random_forest_mapper(
                charset, 
                tile_size,
                train=auto_train_rf and not rf_model_path,
                model_path=rf_model_path,
            )
        
        # Image generator (lazy loaded)
        self._generator: Optional[PromptToImageGenerator] = None
    
    def _get_generator(self) -> PromptToImageGenerator:
        """Get or create the image generator."""
        if self._generator is None:
            self._generator = create_generator(model=self.sd_model)
        return self._generator
    
    def _convert_with_mapper(
        self,
        image: Image.Image,
        mapper: Union[AISSMapper, RandomForestMapper],
        apply_edge_detection: bool = True,
    ) -> str:
        """Convert image to ASCII using specified mapper."""
        return mapper.convert_image(
            image,
            tile_size=self.tile_size,
            apply_edge_detection=apply_edge_detection,
        )
    
    def generate(
        self,
        prompt: str,
        char_width: int = 80,
        char_height: Optional[int] = None,
        apply_edge_detection: bool = True,
        negative_prompt: str = "blurry, low quality, distorted",
        seed: Optional[int] = None,
        return_comparison: bool = False,
    ) -> Union[ASCIIResult, Tuple[ASCIIResult, ASCIIResult]]:
        """
        Generate ASCII art from a text prompt.
        
        Args:
            prompt: Text description of the image
            char_width: Output width in characters
            char_height: Output height in characters (auto if None)
            apply_edge_detection: Apply Canny edges before mapping
            negative_prompt: What to avoid in generation
            seed: Random seed for reproducibility
            return_comparison: Return both AISS and RF results
            
        Returns:
            ASCIIResult (or tuple of results if return_comparison=True)
        """
        start_time = time.time()
        
        # Calculate pixel dimensions
        img_width = char_width * self.tile_size[0]
        img_height = (char_height or 40) * self.tile_size[1]
        
        # Generate image
        generator = self._get_generator()
        source_image = generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=img_width,
            height=img_height,
            seed=seed,
        )
        
        gen_time = time.time() - start_time
        
        # Convert to ASCII
        if return_comparison or self.mapper_type == "both":
            # Compare both mappers
            aiss_start = time.time()
            aiss_ascii = self._convert_with_mapper(
                source_image, 
                self._aiss_mapper or create_aiss_mapper(self.charset_name, self.tile_size),
                apply_edge_detection,
            )
            aiss_time = time.time() - aiss_start
            
            rf_start = time.time()
            rf_ascii = self._convert_with_mapper(
                source_image,
                self._rf_mapper or create_random_forest_mapper(self.charset_name, self.tile_size),
                apply_edge_detection,
            )
            rf_time = time.time() - rf_start
            
            aiss_result = create_result(
                text=aiss_ascii,
                source_image=source_image,
                prompt=prompt,
                mapper="aiss",
                charset=self.charset_name,
                generation_time=f"{gen_time:.2f}s",
                mapping_time=f"{aiss_time:.2f}s",
            )
            
            rf_result = create_result(
                text=rf_ascii,
                source_image=source_image,
                prompt=prompt,
                mapper="random_forest",
                charset=self.charset_name,
                generation_time=f"{gen_time:.2f}s",
                mapping_time=f"{rf_time:.2f}s",
            )
            
            return aiss_result, rf_result
        
        else:
            # Single mapper
            mapper = self._aiss_mapper if self.mapper_type == "aiss" else self._rf_mapper
            map_start = time.time()
            ascii_text = self._convert_with_mapper(source_image, mapper, apply_edge_detection)
            map_time = time.time() - map_start
            
            return create_result(
                text=ascii_text,
                source_image=source_image,
                prompt=prompt,
                mapper=self.mapper_type,
                charset=self.charset_name,
                generation_time=f"{gen_time:.2f}s",
                mapping_time=f"{map_time:.2f}s",
            )
    
    def from_image(
        self,
        image: Union[str, Image.Image],
        char_width: int = 80,
        char_height: Optional[int] = None,
        apply_edge_detection: bool = True,
        return_comparison: bool = False,
    ) -> Union[ASCIIResult, Tuple[ASCIIResult, ASCIIResult]]:
        """
        Convert an existing image to ASCII art.
        
        Args:
            image: PIL Image or path to image file
            char_width: Output width in characters
            char_height: Output height in characters
            apply_edge_detection: Apply Canny edges before mapping
            return_comparison: Return both AISS and RF results
            
        Returns:
            ASCIIResult (or tuple of results)
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)
        
        # Resize for target dimensions
        image = resize_for_ascii(
            image,
            char_width=char_width,
            char_height=char_height,
            tile_size=self.tile_size,
        )
        
        if return_comparison or self.mapper_type == "both":
            aiss_ascii = self._convert_with_mapper(
                image,
                self._aiss_mapper or create_aiss_mapper(self.charset_name, self.tile_size),
                apply_edge_detection,
            )
            rf_ascii = self._convert_with_mapper(
                image,
                self._rf_mapper or create_random_forest_mapper(self.charset_name, self.tile_size),
                apply_edge_detection,
            )
            
            return (
                create_result(text=aiss_ascii, source_image=image, mapper="aiss", charset=self.charset_name),
                create_result(text=rf_ascii, source_image=image, mapper="random_forest", charset=self.charset_name),
            )
        
        else:
            mapper = self._aiss_mapper if self.mapper_type == "aiss" else self._rf_mapper
            ascii_text = self._convert_with_mapper(image, mapper, apply_edge_detection)
            
            return create_result(
                text=ascii_text,
                source_image=image,
                mapper=self.mapper_type,
                charset=self.charset_name,
            )
    
    def compare_results(
        self,
        aiss_result: ASCIIResult,
        rf_result: ASCIIResult,
    ) -> dict:
        """
        Compare AISS and Random Forest results using quality metrics.
        
        Args:
            aiss_result: Result from AISS mapper
            rf_result: Result from Random Forest mapper
            
        Returns:
            Dictionary with comparison metrics
        """
        source = aiss_result.source_image or rf_result.source_image
        if source is None:
            return {"error": "No source image available for comparison"}
        
        return compare_mappers(source, aiss_result.text, rf_result.text)
    
    @staticmethod
    def available_charsets() -> list:
        """List available character sets."""
        return list_charsets()
    
    def save_rf_model(self, path: str):
        """Save trained Random Forest model."""
        if self._rf_mapper:
            self._rf_mapper.save_model(path)
    
    def load_rf_model(self, path: str):
        """Load pre-trained Random Forest model."""
        if self._rf_mapper is None:
            self._rf_mapper = RandomForestMapper(charset=self._charset)
        self._rf_mapper.load_model(path)


# Convenience function for quick usage
def prompt_to_ascii(
    prompt: str,
    mapper: str = "random_forest",
    charset: str = "ascii_standard",
    char_width: int = 80,
    **kwargs
) -> ASCIIResult:
    """
    Quick function to generate ASCII art from a prompt.
    
    Args:
        prompt: Text description
        mapper: "aiss" or "random_forest"
        charset: Character set name
        char_width: Output width in characters
        **kwargs: Additional arguments for PromptToASCII.generate()
        
    Returns:
        ASCIIResult with generated ASCII art
    """
    pipeline = PromptToASCII(mapper=mapper, charset=charset)
    return pipeline.generate(prompt, char_width=char_width, **kwargs)


def image_to_ascii(
    image: Union[str, Image.Image],
    mapper: str = "random_forest",
    charset: str = "ascii_standard",
    char_width: int = 80,
    **kwargs
) -> ASCIIResult:
    """
    Quick function to convert an image to ASCII art.
    
    Args:
        image: PIL Image or path to image
        mapper: "aiss" or "random_forest"
        charset: Character set name
        char_width: Output width in characters
        **kwargs: Additional arguments
        
    Returns:
        ASCIIResult with converted ASCII art
    """
    pipeline = PromptToASCII(mapper=mapper, charset=charset, auto_train_rf=True)
    return pipeline.from_image(image, char_width=char_width, **kwargs)
