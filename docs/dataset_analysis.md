# Dataset Analysis

## Structure
- products: **68** | curated outfits: **25**
- product fields: ['id', 'name', 'brand', 'price_inr', 'rating', 'rating_count', 'gender', 'wear_type', 'category', 'category_label', 'occasion', 'tags', 'description', 'description_source', 'image', 'site', 'product_url', 'collected_at', 'coarse_type', 'image_path']
- outfit fields: ['outfit_id', 'gender', 'wear_type', 'occasion', 'theme', 'hero', 'hero_id', 'second', 'second_id', 'layer', 'layer_id', 'footwear', 'footwear_id', 'accessory_1', 'accessory_1_id', 'accessory_2', 'accessory_2_id', 'palette', 'items_count', 'total_price_inr', 'image_files', 'stylist_rationale']

## Product gender
- women: 41
- men: 27

## Product wear_type
- western: 31
- footwear: 17
- accessory: 13
- ethnic: 7

## Derived coarse type
- footwear: 17
- dress: 13
- accessory: 13
- top: 12
- bottom: 9
- layer: 4

## Product occasion
- casual: 15
- party: 13
- office: 12
- festive: 9
- wedding: 6
- sports: 5
- vacation: 4
- winter: 4

## Outfit occasion
- party: 6
- casual: 5
- festive: 4
- wedding: 3
- office: 2
- vacation: 2
- sports: 2
- winter: 1

## Palette distribution (outfits)
- black / red: 2
- maroon / red / white: 1
- red / white / black: 1
- grey / brown / pink / black: 1
- red / cream / white: 1
- purple / red / white: 1
- navy / green / white / olive: 1
- white / blue / cream / black / brown: 1
- navy / brown / olive: 1
- brown / black: 1
- tan / black / cream: 1
- red / blue / grey / black: 1
- maroon / blue / cream / black: 1
- white / red: 1
- navy / brown / pink: 1
- navy / green / brown / red: 1
- green / navy / olive: 1
- grey / white / black / red: 1
- white / red / black: 1
- grey / brown / green / olive: 1
- black / navy / brown / red: 1
- beige / brown / red: 1
- blue / green / olive: 1
- navy / green / brown: 1

## Compatibility graph
- co-occurrence pairs: 124
- items used in outfits: 68 / 68
- items appearing in >1 outfit: 19 (sparsity → fuse with embeddings)

## Outfit role fill rate
- hero: 25/25
- second: 13/25
- layer: 4/25
- footwear: 25/25
- accessory_1: 22/25
- accessory_2: 4/25

## Data quality notes
- No per-item colour field; colour lives in outfit `palette` and in product names/descriptions. Colour reasoning is therefore drawn from the outfit palette / LLM, not a structured field.
- Missing per column: rating=25, rating_count=42

## Images
- product rows: 68 | missing image files: 0
