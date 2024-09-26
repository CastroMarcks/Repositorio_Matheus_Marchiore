SELECT   
    funil.jobid,
    funil.SubscriberKey,
    sub.cnpj
FROM 
    ent.de_20240731_cdo_dashboard_funil_de_conversao AS funil
LEFT JOIN 
    (
       select
per.email,
catt.attribute_value as cnpj,
cus.subscription_id

from
ent.SellersV2_customer_attributes as catt
join ent.SellersV2_person_customer as cus on cus.customer_id = catt.customer_id
join ent.SellersV2_person as per on per.person_id = cus.person_id

where
catt.attribute_name = 'cnpj'
and catt.attribute_value is not null) AS sub 
ON 
    sub.subscription_id = funil.SubscriberKey
    
where sub.cnpj is not null