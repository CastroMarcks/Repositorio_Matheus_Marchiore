select
    envio.jobid,
    envio.SubscriberKey,
    envio.data_envio_format,
    b.fromname as remetente,
    b.fromemail as email_remetente,
    b.emailname as nome_email,
    abertura.data_abertura_format,
    bounce.data_bounce_format,
    clique.data_clique_format,
    cadastro.data_ini_cad_cdo_format,
    cadastro.data_updt_cad_cdo_format,
    won.data_won_cdo_format,
    sub.cnpj
from
    (
        select
            jobid,
            SubscriberKey,
            format(min(EventDate), 'yyyy-MM-dd') as data_envio_format
        from
            _Sent
        group by
            jobid,
            SubscriberKey
    ) as envio
    left join _job as b on envio.jobid = b.jobid
    left join (
        select
            jobid,
            subscriberkey,
            format(min(eventdate), 'yyyy-MM-dd') as data_abertura_format
        from
            _open
        group by
            jobid,
            subscriberkey
    ) as abertura on envio.jobid = abertura.jobid and envio.SubscriberKey = abertura.subscriberkey
    left join (
        select
            jobid,
            subscriberkey,
            format(min(eventdate), 'yyyy-MM-dd') as data_bounce_format
        from
            _bounce
        group by
            jobid,
            subscriberkey
    ) as bounce on envio.jobid = bounce.jobid and envio.SubscriberKey = bounce.subscriberkey
    left join (
        select
            jobid,
            subscriberkey,
            format(min(eventdate), 'yyyy-MM-dd') as data_clique_format
        from
            _click
        group by
            jobid,
            subscriberkey
    ) as clique on envio.jobid = clique.jobid and envio.SubscriberKey = clique.subscriberkey
    left join (
        select
            lead_id as subscriberkey,
            format(created_at, 'yyyy-MM-dd') as data_ini_cad_cdo_format,
            format(updated_at, 'yyyy-MM-dd') as data_updt_cad_cdo_format
        from
            ent.leadsv2_attributes
        where
            attribute_name = 'onboarding_step'
            and value is not null
            and product = 'conta digital'
    ) as cadastro on envio.SubscriberKey = cadastro.subscriberkey
    left join (
        select
            lead_id as subscriberkey,
            format(created_at, 'yyyy-MM-dd') as data_won_cdo_format
        from
            ent.leadsv2_attributes
        where
            attribute_name = 'status'
            and value = 'won'
            and product = 'conta digital'
    ) as won on envio.SubscriberKey = won.subscriberkey
    left join (
        select
            cus.subscription_id,
            catt.attribute_value as cnpj
        from
            ent.SellersV2_customer_attributes as catt
            join ent.SellersV2_person_customer as cus on cus.customer_id = catt.customer_id
            join ent.SellersV2_person as per on per.person_id = cus.person_id
        where
            catt.attribute_name = 'cnpj'
            and catt.attribute_value is not null
    ) as sub on sub.subscription_id = envio.SubscriberKey
