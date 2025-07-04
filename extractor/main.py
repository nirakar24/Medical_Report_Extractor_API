from .extractors.cbc import extract_cbc_from_image
from .extractors.lft import extract_lft_from_image

def extract_report(report_type, image_path_or_bytes):
    if report_type == 'cbc':
        return extract_cbc_from_image(image_path_or_bytes)
    elif report_type == 'lft':
        return extract_lft_from_image(image_path_or_bytes)
    else:
        return {'error': f'Unknown report type: {report_type}'} 