SELECT 
    n.id as Negociation_id_bd, 
    n.created_at as date_contact, 
    n.updated_at, 
    n.seller_id, 
    n.status, 
    n.phone_number, 
    n.sender_phone, 
    n.strategy as strategy_tag, 
    m.slug AS model_tag, 
    n.extra_info, 
    n.region
FROM 
    datalake_silver.communicator_api_negotiations_negotiation n
JOIN 
    datalake_silver.communicator_api_negotiations_negotiationmodel m 
ON 
    n.model_id = m.id
WHERE 
    MONTH(n.created_at) = {mes} 
    AND n.strategy = '{strategy}'
