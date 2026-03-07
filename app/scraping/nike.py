"""
Nike product scraper: extracts product info, sizing charts,
and material composition from Nike.com product pages.
"""

import re
import httpx
from bs4 import BeautifulSoup

from app.models.schemas import GarmentInfo, GarmentSize, FitType


NIKE_API_BASE = "https://api.nike.com"
NIKE_PRODUCT_URL = "https://www.nike.com"

# Nike uses a public GraphQL/REST API for product data
NIKE_PRODUCT_API = f"{NIKE_API_BASE}/cic/browse/v2"
NIKE_SIZE_API = f"{NIKE_API_BASE}/deliver/available_skus/v1"


async def fetch_product_page(url: str) -> dict:
    """
    Fetch product data from Nike.com.
    Nike embeds product JSON in the page as __NEXT_DATA__ or via API.
    """
    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                          "Mobile/15E148 Safari/604.1",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Nike embeds product data in a __NEXT_DATA__ script tag
        next_data = soup.find("script", id="__NEXT_DATA__")
        if next_data:
            import json
            data = json.loads(next_data.string)
            result = extract_from_next_data(data)
            # If no images from JSON, try extracting from HTML
            if not result.get("image_urls"):
                result["image_urls"] = extract_images_from_soup(soup)
            return result

        # Fallback: parse HTML directly
        return parse_product_html(soup)


def extract_from_next_data(data: dict) -> dict:
    """Extract product info from Nike's __NEXT_DATA__ JSON."""
    try:
        props = data.get("props", {}).get("pageProps", {})
        product_data = (
            props.get("initialState", {})
            .get("product", {})
        )

        # Navigate Nike's nested product structure
        products = product_data.get("products", {})
        if not products:
            # Try alternative path
            product_data = props.get("productInfo", {})
            return _extract_product_info(product_data)

        product_id = next(iter(products))
        product = products[product_id]
        return _extract_product_info(product)

    except (KeyError, StopIteration):
        return {}


def _extract_product_info(product: dict) -> dict:
    """Normalize Nike product data into our format."""
    image_urls = []

    # Try multiple image paths in Nike's data
    images = product.get("images", {})
    if isinstance(images, dict):
        for key in ["squarishURL", "portraitURL", "squarish", "portrait"]:
            val = images.get(key)
            if isinstance(val, list):
                for img in val:
                    url = img.get("url", "") if isinstance(img, dict) else str(img)
                    if url and url not in image_urls:
                        image_urls.append(url)
            elif isinstance(val, str) and val:
                image_urls.append(val)
    elif isinstance(images, list):
        for img in images:
            url = img.get("url", "") if isinstance(img, dict) else str(img)
            if url and url not in image_urls:
                image_urls.append(url)

    # Try nodes.nodes pattern
    nodes = product.get("nodes", product.get("colorwayImages", []))
    if isinstance(nodes, list):
        for node in nodes:
            url = node.get("squarishURL", node.get("portraitURL", ""))
            if url and url not in image_urls:
                image_urls.append(url)

    return {
        "product_id": product.get("id", product.get("styleColor", "")),
        "name": product.get("title", product.get("name", "")),
        "url": product.get("url", ""),
        "image_urls": image_urls,
        "description": product.get("description", ""),
        "fit_type": product.get("fitType", "regular"),
        "material": product.get("descriptionPreview", ""),
        "color": product.get("colorDescription", ""),
        "sizes": product.get("availableSkus", []),
    }


def extract_images_from_soup(soup: BeautifulSoup) -> list[str]:
    """Extract product images from Nike page using multiple strategies."""
    images = []

    # Strategy 1: og:image meta tag (most reliable)
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        images.append(og_image["content"])

    # Strategy 2: twitter:image meta tag
    tw_image = soup.find("meta", attrs={"name": "twitter:image"})
    if tw_image and tw_image.get("content"):
        url = tw_image["content"]
        if url not in images:
            images.append(url)

    # Strategy 3: Find all Nike CDN image URLs in the page
    for img in soup.find_all("img"):
        src = img.get("src", "") or img.get("data-src", "")
        if "nike.com" in src and ("product" in src or "i1" in src or "static.nike.com" in src):
            if src not in images and src.startswith("http"):
                images.append(src)

    # Strategy 4: Search for Nike image URLs in all script tags
    for script in soup.find_all("script"):
        if script.string:
            urls = re.findall(
                r'https?://[^"\'\s]+\.nike\.com/[^"\'\s]+\.(?:jpg|jpeg|png|webp)',
                script.string,
            )
            for url in urls:
                clean_url = url.split("?")[0]
                if clean_url not in images and len(images) < 10:
                    images.append(clean_url)

    # Upgrade any image URLs to highest resolution variant
    upgraded = []
    for url in images:
        upgraded_url = _upgrade_nike_image_url(url)
        if upgraded_url not in upgraded:
            upgraded.append(upgraded_url)

    return upgraded


def _upgrade_nike_image_url(url: str) -> str:
    """
    Upgrade a Nike CDN image URL to the highest resolution version.
    Nike uses URL patterns like /t_PDP_1728_v1/ for high-res images.
    """
    # Replace any resolution token with the highest available
    res_patterns = [
        r"t_PDP_\d+_v\d+", r"t_PDP_\d+", r"t_default",
        r"t_web_pdp_\d+", r"t_product_v1",
    ]
    for pattern in res_patterns:
        if re.search(pattern, url):
            return re.sub(pattern, "t_PDP_1728_v1", url)
    return url


def parse_product_html(soup: BeautifulSoup) -> dict:
    """Fallback HTML parser for Nike product pages."""
    title = soup.find("h1", {"id": "pdp_product_title"})
    if not title:
        title = soup.find("h1")
    description = soup.find("div", {"class": re.compile("description")})

    images = extract_images_from_soup(soup)

    return {
        "name": title.text.strip() if title else "",
        "description": description.text.strip() if description else "",
        "image_urls": images,
    }


def parse_material_composition(description: str) -> dict[str, float]:
    """
    Parse material composition from product description.
    E.g. "Body: 88% polyester/12% elastane" -> {"polyester": 0.88, "elastane": 0.12}
    """
    composition = {}
    pattern = r"(\d+)%\s*(\w+)"
    matches = re.findall(pattern, description.lower())

    for pct, material in matches:
        material = material.strip()
        composition[material] = float(pct) / 100.0

    # Normalize to sum to 1.0
    total = sum(composition.values())
    if total > 0:
        composition = {k: round(v / total, 3) for k, v in composition.items()}

    return composition


def infer_fit_type(description: str, name: str) -> FitType:
    """Infer garment fit type from description and name."""
    text = (description + " " + name).lower()

    if any(kw in text for kw in ["slim", "tight", "fitted", "compression"]):
        return FitType.slim
    if any(kw in text for kw in ["oversized", "oversize", "boxy"]):
        return FitType.oversized
    if any(kw in text for kw in ["relaxed", "loose", "easy"]):
        return FitType.relaxed
    return FitType.regular


# Nike size charts for tops (in cm) - common across most Nike tops
# These are the actual garment measurements, not body measurements
NIKE_MENS_TOP_SIZES: dict[str, dict] = {
    "XS": {"chest_cm": 88.0, "length_cm": 68.0, "shoulder_cm": 42.0, "sleeve_cm": 59.0},
    "S":  {"chest_cm": 93.0, "length_cm": 70.0, "shoulder_cm": 44.0, "sleeve_cm": 61.0},
    "M":  {"chest_cm": 100.0, "length_cm": 72.0, "shoulder_cm": 47.0, "sleeve_cm": 63.0},
    "L":  {"chest_cm": 108.0, "length_cm": 74.0, "shoulder_cm": 50.0, "sleeve_cm": 65.0},
    "XL": {"chest_cm": 116.0, "length_cm": 76.0, "shoulder_cm": 53.0, "sleeve_cm": 67.0},
    "2XL": {"chest_cm": 124.0, "length_cm": 78.0, "shoulder_cm": 56.0, "sleeve_cm": 69.0},
}

NIKE_WOMENS_TOP_SIZES: dict[str, dict] = {
    "XS": {"chest_cm": 82.0, "length_cm": 60.0, "shoulder_cm": 37.0, "sleeve_cm": 54.0},
    "S":  {"chest_cm": 87.0, "length_cm": 62.0, "shoulder_cm": 39.0, "sleeve_cm": 56.0},
    "M":  {"chest_cm": 93.0, "length_cm": 64.0, "shoulder_cm": 41.0, "sleeve_cm": 58.0},
    "L":  {"chest_cm": 100.0, "length_cm": 66.0, "shoulder_cm": 44.0, "sleeve_cm": 60.0},
    "XL": {"chest_cm": 108.0, "length_cm": 68.0, "shoulder_cm": 47.0, "sleeve_cm": 62.0},
    "2XL": {"chest_cm": 116.0, "length_cm": 70.0, "shoulder_cm": 50.0, "sleeve_cm": 64.0},
}


def get_nike_sizes(gender: str = "male") -> list[GarmentSize]:
    """Return Nike's standard sizing chart as GarmentSize objects."""
    chart = NIKE_MENS_TOP_SIZES if gender == "male" else NIKE_WOMENS_TOP_SIZES
    return [
        GarmentSize(size_label=label, **dims)
        for label, dims in chart.items()
    ]


async def scrape_nike_product(url: str, gender: str = "male") -> GarmentInfo:
    """
    Full scraping pipeline for a Nike product URL.
    Returns structured GarmentInfo.
    """
    raw_data = await fetch_product_page(url)

    material_text = raw_data.get("material", raw_data.get("description", ""))
    materials = parse_material_composition(material_text)
    if not materials:
        materials = {"polyester": 1.0}  # default assumption

    fit = infer_fit_type(
        raw_data.get("description", ""),
        raw_data.get("name", ""),
    )

    sizes = get_nike_sizes(gender)

    return GarmentInfo(
        product_id=raw_data.get("product_id", url.split("/")[-1]),
        name=raw_data.get("name", "Unknown Nike Product"),
        brand="Nike",
        url=url,
        image_urls=raw_data.get("image_urls", []),
        fit_type=fit,
        material_composition=materials,
        sizes=sizes,
        category="top",
        color=raw_data.get("color", ""),
    )
