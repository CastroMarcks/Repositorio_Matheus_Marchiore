WITH
produtoFinal AS (
    WITH 
    calendar AS (
        SELECT cast(dia as date) dia
        FROM datalake_silver.operations_orders_bi_ops_datas
        WHERE cast(dia as date) > date_add('day', -150, cast(at_timezone(current_date,'America/Sao_Paulo') as timestamp))
        AND cast(dia as date) <= cast(at_timezone(current_date,'America/Sao_Paulo') as timestamp)
    ),
    products AS (
        SELECT DISTINCT
            product_id as sku
        FROM datalake_gold.bio_product
    ),
    baseFinal AS (
        SELECT 
            c.dia,
            p.sku
        FROM calendar AS c
        CROSS JOIN products AS p
    ),
    stockNRow AS (
        SELECT (at_timezone(sh.updated_at,'America/Sao_Paulo')) as updated_at,
            sh.seller_product_id,
            bp.product_id as sku,
            sh.quantity,
            row_number() over(partition by sh.seller_product_id, cast(at_timezone(sh.updated_at,'America/Sao_Paulo') as date) order by at_timezone(sh.updated_at,'America/Sao_Paulo') desc) as nrow
        FROM datalake_silver.products_api_seller_products_stockhistory as sh
        INNER JOIN datalake_gold.bio_product as bp
            ON bp.seller_product_id = sh.seller_product_id 
        WHERE at_timezone(sh.updated_at,'America/Sao_Paulo') >= date_add('day', -120, at_timezone(current_timestamp,'America/Sao_Paulo'))
    ),
    maxStock AS (
        SELECT 
            DATE_FORMAT(dia, '%Y-%m') AS mes_ano,
            sku,
            MAX(pf.stock) AS max_stock
        FROM produtoFinal AS pf
        INNER JOIN datalake_gold.bio_product AS bp ON bp.product_id = pf.sku
        WHERE sku IN ({formatted_skus})
        AND dia > DATE('{date}')
        AND MONTH(dia) = MONTH(DATE('{date}'))
        AND YEAR(dia) = YEAR(DATE('{date}'))
        GROUP BY DATE_FORMAT(dia, '%Y-%m'), sku
    ),
    maxStockDay AS (
        SELECT
            pf.dia,
            pf.sku
        FROM
            produtoFinal AS pf
            INNER JOIN maxStock AS ms ON pf.sku = ms.sku AND pf.stock = ms.max_stock
        WHERE dia > DATE('{date}') 
        AND MONTH(dia) = MONTH(DATE('{date}'))
        AND YEAR(dia) = YEAR(DATE('{date}'))
    )
SELECT
    DATE_FORMAT(msd.dia, '%Y-%m') AS mes_ano,
    msd.sku,
    ms.max_stock AS quantity_pos_disparo,
    msd.dia as aumentou
FROM maxStockDay AS msd
INNER JOIN maxStock AS ms ON msd.sku = ms.sku
ORDER BY quantity_pos_disparo DESC
