WITH financial_control AS (    
    SELECT
        seller_order_item_code,
        DATE_TRUNC('week', accounted_at) AS semana_competencia,
        DATE_TRUNC('month', accounted_at) AS mes_competencia,
        CAST(SUM(
            CASE
                WHEN provision_type IN ('seller_transfer', 'seller_transfer_chargeback') THEN -1 * relative_amount
            END) AS DECIMAL(24,2)) AS Gmv,
        coalesce(CAST(SUM(
            CASE
                WHEN provision_type IN ('seller_commission', 'seller_commission_chargeback', 'seller_commission_fine', 'seller_commission_fine_chargeback') THEN -1 * relative_amount
            ELSE 0
            END) * (1 - 0.1125) AS DECIMAL(24,2)),0) AS Commission_net,
        coalesce(CAST(SUM(
            CASE
                WHEN provision_type IN ('seller_flat_fee', 'seller_flat_fee_chargeback', 'seller_flat_fee_fine', 'seller_flat_fee_fine_chargeback') THEN -1 * relative_amount
            ELSE 0
            END) * (1 - 0.1125) AS DECIMAL(24,2)),0) AS Flat_fee_net,
        coalesce(CAST(SUM(
            CASE
                WHEN provision_type IN ('seller_markup', 'seller_markup_chargeback') THEN -1 * relative_amount
            ELSE 0
            END) * (1 - 0.1125) AS DECIMAL(24,2)),0) AS Revenue_markup_net,
        coalesce(CAST(SUM(
            CASE
                WHEN provision_type IN ('seller_subscription', 'seller_subscription_chargeback') THEN -1 * relative_amount
            ELSE 0
            END) * (1 - 0.0565) AS DECIMAL(24,2)),0) AS Subscription_net,
        coalesce(CAST(SUM(
            CASE
                WHEN provision_type IN ('marketplace_commission_discount', 'marketplace_commission', 'marketplace_commission_chargeback',
                                        'marketplace_commission_fine', 'marketplace_commission_fine_chargeback',
                                        'marketplace_flat_fee', 'marketplace_flat_fee_chargeback',
                                        'marketplace_flat_fee_fine', 'marketplace_flat_fee_fine_chargeback') THEN relative_amount
            ELSE 0
            END) * (1 - 0.0925) AS DECIMAL(24,2)) * -1,0) AS Net_COGS,
        coalesce(CAST(SUM(
            CASE
                WHEN provision_type IN ('seller_incentive_value', 'seller_incentive_value_chargeback', 'seller_subsidy', 'seller_subsidy_chargeback', 'seller_price_discount', 'seller_price_discount_chargeback', 'marketplace_subsidy', 'marketplace_subsidy_chargeback', 'seller_flat_freight_reduced', 'seller_flat_freight_reduced_chargeback', 'seller_freight_reduced', 'seller_freight_reduced_chargeback', 'seller_markup_incentive', 'seller_markup_incentive_chargeback', 'seller_operation_incentive', 'seller_operation_incentive_chargeback') THEN -1 * relative_amount
            ELSE 0
            END) AS DECIMAL(24,2)),0) AS Sales_incentive_wihtout_ads,
        coalesce(CAST(SUM(
            CASE
                WHEN provision_type IN ('seller_flat_freight_deduction', 'seller_flat_freight_deduction_chargeback',
                                        'seller_freight_buyer_deduction', 'seller_freight_buyer_deduction_chargeback',
                                        'seller_freight_increased', 'seller_freight_increased_chargeback',
                                        'carrier_quoted', 'carrier_quoted_chargeback', 'carrier_quoted_adjustment',
                                        'driver_first_mile', 'driver_complements_first_mile') THEN -1 * relative_amount
            ELSE 0
            END) AS DECIMAL(24,2)),0) AS freight_result
    FROM datalake_silver.controller_api_accountingsellerstore_accountingsellerstore
    GROUP BY seller_order_item_code, DATE_TRUNC('week', accounted_at), DATE_TRUNC('month', accounted_at)
)
SELECT
    bio.seller_order_item_id,
    bio.seller_order_item_code,
    ROUND(SUM(fc.Commission_net + fc.Flat_fee_net + fc.Revenue_markup_net + fc.Subscription_net + fc.Net_COGS + fc.Sales_incentive_wihtout_ads + fc.freight_result), 2) AS gross_profit_adjusted
FROM datalake_gold.bio_orderitem bio
LEFT JOIN financial_control fc ON bio.seller_order_item_code = fc.seller_order_item_code
WHERE bio.seller_order_item_id in ({formatted_ids})
GROUP BY bio.seller_order_item_id, bio.seller_order_item_code
