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
            on bp.seller_product_id = sh.seller_product_id 
        WHERE at_timezone(sh.updated_at,'America/Sao_Paulo') >= date_add('day', -120, at_timezone(current_timestamp,'America/Sao_Paulo'))
    ),
    purchase AS (
        SELECT
            cast((at_timezone(purchase_timestamp, 'America/Sao_Paulo')) as date) as purchase_date,
            product_id as sku,
            round(sum(quantity)) as quantity,
            round(sum(gmv), 2) as gmv
        FROM datalake_gold.bio_orderitem 
        WHERE at_timezone(purchase_timestamp, 'America/Sao_Paulo') >= date_add('day', -120, at_timezone(current_timestamp,'America/Sao_Paulo')) 
        GROUP BY 1, 2
    ),
    stockBefore60 AS (
        SELECT
            date_add('day', -150, at_timezone(current_timestamp,'America/Sao_Paulo')) as updated_at,
            sh.seller_product_id,
            bp.product_id as sku,
            max_by(quantity, at_timezone(sh.updated_at,'America/Sao_Paulo')) as quantity,
            0 as nrow
        FROM datalake_silver.products_api_seller_products_stockhistory as sh
        INNER JOIN datalake_gold.bio_product as bp 
            on bp.seller_product_id = sh.seller_product_id 
        WHERE at_timezone(sh.updated_at,'America/Sao_Paulo') < date_add('day', -150, at_timezone(current_timestamp,'America/Sao_Paulo'))
        GROUP BY 1, 2, 3
    ),
    latestStock AS (
        SELECT * 
        FROM stockNRow 
        WHERE nrow = 1 
        UNION 
        SELECT * 
        FROM stockBefore60
    ),
    prodInfo AS (
        SELECT  
            cast(updated_at at time zone 'America/Sao_Paulo' as date) as dateStart,
            sku,
            quantity as stock
        FROM latestStock
    ),
    periodGap AS (
        SELECT
            dateStart,
            sku,
            stock as estoque,
            coalesce(lag(dateStart, 1) over(partition by sku order by dateStart desc), cast(date_add('day', 1, at_timezone(current_timestamp,'America/Sao_Paulo')) as date)) as endDate
        FROM prodInfo
    )
    SELECT
        bf.dia,
        bf.sku,
        pg.estoque as stock,
        coalesce(prc.quantity, 0) quantity,
        coalesce (prc.gmv, 0) gmv
    FROM baseFinal as bf
    JOIN periodGap as pg 
        ON bf.dia >= pg.dateStart and bf.dia < pg.endDate and bf.sku = pg.sku
    LEFT JOIN purchase as prc 
        ON prc.sku = bf.sku and prc.purchase_date = bf.dia
    ORDER BY 1 ASC
)
SELECT 
    dia,
    date_format(dia, '%Y-%m') as mes_ano,
    sku as product_id,
    pf.stock as estoque_pre
FROM produtoFinal as pF
INNER JOIN datalake_gold.bio_product as bp ON bp.product_id = pF.sku
WHERE sku in ({formatted_skus})
AND dia = date('{date}') - INTERVAL '1' DAY
ORDER BY dia DESC
