SELECT DISTINCT
    oi.product_id,
    date_format(cast(oi.purchase_timestamp as date), '%Y-%m-%d') as purchase_timestamp,
    bp.full_name as produto,
    bp.seller_id as seller_id,
    oi.seller_order_item_id as order_id,
    oi.freight_value AS freight_value,
    oi.price AS pd_price
FROM datalake_gold.bio_orderitem as oi 
LEFT JOIN datalake_gold.bio_product as bp on bp.product_id = oi.product_id 
WHERE oi.product_id in ({formatted_skus})
AND oi.purchase_timestamp >= DATE('{date}')
AND oi.purchase_timestamp < date_trunc('month', DATE('{date}') + interval '3' month)
AND oi.cancelation_status = ''
AND oi.region = 'br'
AND oi.status_seller_order <> 'pending'
