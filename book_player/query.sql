with tb_lobby as (
    select
        *
    from
        tb_lobby_stats_player
    where
        dtCreatedAt < '2022-02-01'
        and dtCreatedAt > date('2022-02-01', '-30 days')
),
tb_stats as (
    select
        idPlayer,
        count(DISTINCT idLobbyGame) as qtPartidas,
        count(
            DISTINCT case
                when qtRoundsPlayed < 16 then idLobbyGame
            end
        ) as qtPartidasMenos16,
        count(DISTINCT date(dtCreatedAt)) as qtDias,
        1.0 * count(DISTINCT idLobbyGame) / count(DISTINCT date(dtCreatedAt)) as mediaPartidasDia,
        avg(qtKill) as avgqtKill,
        avg(1.0 * (qtKill + qtAssist) / qtDeath) as avgKDA,
        1.0 * sum(qtKill + qtAssist) / sum(qtDeath) as KDAgeral,
        avg(1.0 * (qtKill + qtAssist) / qtRoundsPlayed) as avgKAST,
        1.0 * sum(qtKill + qtAssist) / sum(qtRoundsPlayed) as KARoundGeral,
        avg(qtAssist) as avgqtAssist,
        avg(qtDeath) as avgqtDeath,
        avg(qtHs) as avgqtHs,
        1.0 * sum(qtHs) / sum(qtKill) as txHsGeral,
        avg(1.0 * qtHs / qtKill) as avgHsRate,
        avg(qtBombeDefuse) as avgqtBombeDefuse,
        avg(qtBombePlant) as avgqtBombePlant,
        avg(qtTk) as avgqtTk,
        avg(qtTkAssist) as avgqtTkAssist,
        avg(qt1Kill) as avgqt1Kill,
        avg(qt2Kill) as avgqt2Kill,
        avg(qt3Kill) as avgqt3Kill,
        avg(qt4Kill) as avgqt4Kill,
        avg(qt5Kill) as avgqt5Kill,
        sum(qt4Kill) as sumqt4Kill,
        sum(qt5Kill) as sumqt5Kill,
        avg(qtPlusKill) as avgqtPlusKill,
        avg(qtFirstKill) as avgqtFirstKill,
        avg(vlDamage) as avgvlDamage,
        avg(1.0 * vlDamage / qtRoundsPlayed) as avgDamageRound,
        1.0 * sum(vlDamage) / sum(qtRoundsPlayed) as DamageRoundGeral,
        avg(qtHits) as avgqtHits,
        avg(qtShots) as avgqtShots,
        avg(qtLastAlive) as avgqtLastAlive,
        avg(qtClutchWon) as avgqtClutchWon,
        avg(qtRoundsPlayed) as avgqtRoundsPlayed,
        avg(vlLevel) as avgvlLevel,
        avg(qtSurvived) as avgqtSurvived,
        avg(qtTrade) as avgqtTrade,
        avg(qtFlashAssist) as avgqtFlashAssist,
        avg(qtHitHeadshot) as avgqtHitHeadshot,
        avg(qtHitChest) as avgqtHitChest,
        avg(qtHitStomach) as avgqtHitStomach,
        avg(qtHitLeftAtm) as avgqtHitLeftAtm,
        avg(qtHitRightArm) as avgqtHitRightArm,
        avg(qtHitLeftLeg) as avgqtHitLeftLeg,
        avg(qtHitRightLeg) as avgqtHitRightLeg,
        avg(flWinner) as avgflWinner,
        count(
            DISTINCT case
                when descMapName = 'de_mirage' then idLobbyGame
            end
        ) as qtmiragePartidas,
        count(
            DISTINCT case
                when descMapName = 'de_nuke' then idLobbyGame
            end
        ) as qtnukePartidas,
        count(
            DISTINCT case
                when descMapName = 'de_inferno' then idLobbyGame
            end
        ) as qtinfernoPartidas,
        count(
            DISTINCT case
                when descMapName = 'de_vertigo' then idLobbyGame
            end
        ) as qtvertigoPartidas,
        count(
            DISTINCT case
                when descMapName = 'de_ancient' then idLobbyGame
            end
        ) as qtancientPartidas,
        count(
            DISTINCT case
                when descMapName = 'de_dust2' then idLobbyGame
            end
        ) as qtdust2Partidas,
        count(
            DISTINCT case
                when descMapName = 'de_train' then idLobbyGame
            end
        ) as qttrainPartidas,
        count(
            DISTINCT case
                when descMapName = 'de_overpass' then idLobbyGame
            end
        ) as qtoverpassPartidas,
        count(
            DISTINCT case
                when descMapName = 'de_mirage'
                and flWinner = 1 then idLobbyGame
            end
        ) as qtmiragePartidasVencedoras,
        count(
            DISTINCT case
                when descMapName = 'de_nuke'
                and flWinner = 1 then idLobbyGame
            end
        ) as qtnukePartidasVencedoras,
        count(
            DISTINCT case
                when descMapName = 'de_inferno'
                and flWinner = 1 then idLobbyGame
            end
        ) as qtinfernoPartidasVencedoras,
        count(
            DISTINCT case
                when descMapName = 'de_vertigo'
                and flWinner = 1 then idLobbyGame
            end
        ) as qtvertigoPartidasVencedoras,
        count(
            DISTINCT case
                when descMapName = 'de_ancient'
                and flWinner = 1 then idLobbyGame
            end
        ) as qtancientPartidasVencedoras,
        count(
            DISTINCT case
                when descMapName = 'de_dust2'
                and flWinner = 1 then idLobbyGame
            end
        ) as qtdust2PartidasVencedoras,
        count(
            DISTINCT case
                when descMapName = 'de_train'
                and flWinner = 1 then idLobbyGame
            end
        ) as qttrainPartidasVencedoras,
        count(
            DISTINCT case
                when descMapName = 'de_overpass'
                and flWinner = 1 then idLobbyGame
            end
        ) as qtoverpassPartidasVencedoras
    from
        tb_lobby
    group by
        idPlayer
),
tb_lvl_atual as (
    select
        idPlayer,
        vlLevel
    from
        (
            SELECT
                idLobbyGame,
                idPlayer,
                vlLevel,
                dtCreatedAt,
                row_number() over (
                    PARTITION by idPlayer
                    order by
                        dtCreatedAt desc
                ) as rn
            from
                tb_lobby
        )
    where
        rn = 1
)
select
    t1.*,
    t2.vlLevel as vlLevalAtual
from
    tb_stats as t1
    left join tb_lvl_atual as t2 on t1.idPlayer = t2.idPlayer