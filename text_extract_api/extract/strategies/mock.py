from extract.extract_result import ExtractResult
from text_extract_api.extract.strategies.strategy import Strategy
from text_extract_api.files.file_formats.file_format import FileFormat

class MockStrategy(Strategy):
    @classmethod
    def name(cls) -> str:
        return "mock"

    def extract_text(self, file_format: FileFormat, language: str = 'en') -> ExtractResult:
        return ExtractResult.from_text(
            f"🚂 Railway Demo Mode\n\n"
            f"File: {file_format.filename}\n"
            f"Type: {file_format.mime_type}\n"
            f"Size: {len(file_format.binary)} bytes\n\n"
            f"OCR processing is disabled in Railway demo mode.\n"
            f"To enable full functionality, please:\n"
            f"1. Deploy to a platform with more resources (Vast.ai, DigitalOcean)\n"
            f"2. Or upgrade to Railway Pro with more memory\n\n"
            f"This API is working correctly - just without the heavy ML models."
        )
