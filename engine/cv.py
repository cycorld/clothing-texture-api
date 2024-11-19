import cv2
import numpy as np
import argparse
import os
import time

def get_mask_path(product_path):
    """
    product 이미지 경로로부터 mask 이미지 경로를 생성
    """
    dir_name = os.path.dirname(product_path)
    base_name = os.path.splitext(os.path.basename(product_path))[0]
    return os.path.join(dir_name, f"{base_name}-mask.jpg")

def extract_center_texture(clothing_image, debug=False):
    start_time = time.time()
    """
    Extract texture from the center 50% area of the clothing image
    """
    height, width = clothing_image.shape[:2]
    
    # 중앙 50% 영역 계산
    start_x = width // 4
    end_x = width * 3 // 4
    start_y = height // 4
    end_y = height * 3 // 4
    
    # 중앙 영역 마스크 생성
    center_mask = np.zeros((height, width), dtype=np.uint8)
    center_mask[start_y:end_y, start_x:end_x] = 255
    
    # 텍스처 추출 (중앙 영역만) 및 크롭
    texture = cv2.bitwise_and(clothing_image, clothing_image, mask=center_mask)
    texture = texture[start_y:end_y, start_x:end_x]
    
    # 디버깅을 위한 이미지 저장
    if debug:
        cv2.imwrite('debug_01_center_mask.png', center_mask)
        cv2.imwrite('debug_02_extracted_texture.png', texture)
    
    if debug:
        print(f"텍스처 추출 시간: {time.time() - start_time:.3f}초")
    return texture

def apply_texture_to_white_areas(texture, origin_mask, product_image, debug=False):
    start_time = time.time()
    """
    Apply stretched texture to all areas except black regions in the mask, preserving the brightness of white and gray areas
    """
    # 원본 텍스처 저장
    if debug:
        cv2.imwrite('debug_03_original_texture.png', texture)
    
    # 텍스처를 product 이미지 크기로 리사이즈
    resize_start = time.time()
    stretched_texture = cv2.resize(texture, (product_image.shape[1], product_image.shape[0]))
    if debug:
        print(f"텍스처 리사이즈 시간: {time.time() - resize_start:.3f}초")
    
    # 리사이즈된 텍스처 저장
    if debug:
        cv2.imwrite('debug_04_stretched_texture.png', stretched_texture)
    
    # 마스크가 이미 그레이스케일인지 확인하고 처리
    mask_start = time.time()
    if len(origin_mask.shape) == 3:
        mask = cv2.cvtColor(origin_mask, cv2.COLOR_BGR2GRAY)
    else:
        mask = origin_mask.copy()
    white_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if debug:
        print(f"마스크 처리 시간: {time.time() - mask_start:.3f}초")
    
    # 디버깅을 위해 중간 마스크 저장
    if debug:
        cv2.imwrite('debug_05_threshold_mask.png', mask)
    
    # 텍스처 적용 시간 측정
    apply_start = time.time()
    texture_crop = cv2.bitwise_and(stretched_texture, stretched_texture, mask=mask)
    # 원본 마스크의 색 대비를 더 극적으로 조정
    contrast_mask = cv2.convertScaleAbs(origin_mask, alpha=1.5, beta=0)
    if debug:
        cv2.imwrite('debug_06_contrast_mask.png', contrast_mask)
    texture_applied = cv2.addWeighted(texture_crop, 0.9, origin_mask, 0.1, 0)
    # texture_applied도 BGRA로 변환
    texture_applied_rgba = cv2.cvtColor(texture_applied, cv2.COLOR_BGR2BGRA)
    if debug:
        print(f"텍스처 적용 시간: {time.time() - apply_start:.3f}초")
    
    adjusted_gray = cv2.equalizeHist(white_mask)
    if debug:
        cv2.imwrite('debug_07_adjusted_gray.png', adjusted_gray)

    inverted_mask = cv2.bitwise_not(white_mask)
    if debug:
        cv2.imwrite('debug_08_inverted_mask.png', inverted_mask)  # 디버깅용 이미지 저장
    background = cv2.bitwise_and(product_image, product_image, mask=inverted_mask)
    # 검은색을 투명으로 처리
    # BGRA 형식으로 변환 (알파 채널 추가)
    background_rgba = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    # 검은색 픽셀을 더 정확하게 감지하고 투명하게 설정
    # 약간의 오차를 허용하여 거의 검은색인 픽셀도 포함
    threshold = 10  # 검은색 판단 임계값
    black_pixels = np.where(
        (background_rgba[:, :, 0] <= threshold) & 
        (background_rgba[:, :, 1] <= threshold) & 
        (background_rgba[:, :, 2] <= threshold)
    )
    background_rgba[black_pixels[0], black_pixels[1], 3] = 0
    if debug:
        cv2.imwrite('debug_09_background.png', background)  # 디버깅용 이미지 저장

    if debug:
        cv2.imwrite('debug_10_texture_applied.png', texture_applied)  # 디버깅용 이미지 저장
    # BGRA 형식으로 결과 합성
    result = cv2.add(background_rgba, texture_applied_rgba)
    if debug:
        cv2.imwrite('debug_11_semi_result.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # 디버깅용 이미지 저장
    
    if debug:
        print(f"전체 텍스처 적용 시간: {time.time() - start_time:.3f}초")
        cv2.imwrite('debug_12_last_result.png', result)
    
    # 배경에서 검은색을 투명으로 처리
    return result  # RGBA 이미지 반환

def process_image(clothing_path, product_path, output_path=None, debug=False):
    """
    이미지 처리를 위한 API 함수
    
    Args:
        clothing_path (str): 의류 텍스처 이미지 경로
        product_path (str): 제품 이미지 경로
        output_path (str, optional): 결과 이미지 저장 경로
        debug (bool): 디버그 모드 활성화 여부
    
    Returns:
        numpy.ndarray: RGBA 형식의 결과 이미지
    """
    try:
        # 이미지 로드
        clothing_image = cv2.imread(clothing_path)
        product_image = cv2.imread(product_path)
        mask_path = get_mask_path(product_path)
        mask_image = cv2.imread(mask_path)
        
        # 이미지 유효성 검사
        if clothing_image is None:
            raise ValueError(f"의류 이미지를 불러올 수 없습니다: {clothing_path}")
        if product_image is None:
            raise ValueError(f"제품 이미지를 불러올 수 없습니다: {product_path}")
        if mask_image is None:
            raise ValueError(f"마스크 이미지를 불러올 수 없습니다: {mask_path}")
            
        # 텍스처 추출 및 적용
        texture = extract_center_texture(clothing_image, debug=debug)
        result_image = apply_texture_to_white_areas(texture, mask_image, product_image, debug=debug)
        
        # 결과 저장
        if output_path:
            cv2.imwrite(output_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
        return result_image
        
    except Exception as e:
        raise Exception(f"이미지 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply texture from clothing to product image')
    parser.add_argument('clothing_path', type=str, help='Path to the clothing texture image')
    parser.add_argument('product_path', type=str, help='Path to the product image')
    parser.add_argument('--output', type=str, default='result.png', help='Path for the output image (default: result.png)')
    parser.add_argument('--threshold', type=int, default=105, help='Threshold for texture extraction (default: 105)')
    parser.add_argument('--debug', action='store_true', help='Save debug images during processing')
    
    args = parser.parse_args()

    # 이미지 로드
    clothing_image = cv2.imread(args.clothing_path)
    product_image = cv2.imread(args.product_path)
    mask_path = get_mask_path(args.product_path)
    mask_image = cv2.imread(mask_path)
    
    if clothing_image is None:
        raise ValueError(f"Could not load clothing image from {args.clothing_path}")
    if product_image is None:
        raise ValueError(f"Could not load product image from {args.product_path}")
    if mask_image is None:
        raise ValueError(f"Could not load mask image from {mask_path}")

    # 전체 실행 시간 측정
    total_start = time.time()
    
    texture = extract_center_texture(clothing_image, debug=args.debug)
    result_image = apply_texture_to_white_areas(texture, mask_image, product_image, debug=args.debug)
    
    if args.debug:
        print(f"전체 실행 시간: {time.time() - total_start:.3f}초")
    
    # 결과 저장 - PNG 포맷으로 저장하여 투명도 유지
    cv2.imwrite(args.output, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    # 결과 표시 (알파 채널은 제외하고 표시)
    cv2.imshow('Final Image', result_image[:, :, :3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()