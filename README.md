# Projeto de Predição para a GC

## Sobre o Projeto

Este projeto visa desenvolver e implementar algoritmos de predição para resolver problemas reais de negócio na GC. Utilizando as metodologias CRISP-DM para o desenvolvimento e SEMMA para a modelagem dos dados, pretendemos abordar diversas oportunidades de solução.

## Oportunidades de Solução

Dentro deste projeto, identificamos várias oportunidades de solução que abordaremos:

1. **Predição de jogadores que jogarão na próxima semana/mês:**
   - Identificar quais jogadores estarão ativos em determinados períodos.

2. **Predição de churn:**
   - Prever quais jogadores estão propensos a abandonar o jogo.

3. **Predição de assinatura:**
   - Estimar quais usuários têm maior probabilidade de se tornarem assinantes.

4. **Predição de churn de assinatura:**
   - Antecipar quais assinantes podem cancelar suas assinaturas.

## Metodologia

### CRISP-DM

CRISP-DM (Cross Industry Standard Process for Data Mining) é uma metodologia padrão para mineração de dados. As etapas incluem:

1. **Entendimento do Negócio:** Compreensão dos objetivos e requisitos do negócio.
2. **Entendimento dos Dados:** Coleta inicial de dados para familiarização.
3. **Preparação dos Dados:** Limpeza e transformação dos dados para análise.
4. **Modelagem:** Aplicação de técnicas de modelagem para criar modelos preditivos.
5. **Avaliação:** Avaliação dos modelos para garantir que atendem aos objetivos do negócio.
6. **Desdobramento:** Implementação dos modelos em um ambiente de produção.
   
![CRISP-DM](https://miro.medium.com/v2/resize:fit:988/0*tA5OjppLK627FfFo)

### SEMMA

SEMMA (Sample, Explore, Modify, Model, Assess) é uma metodologia desenvolvida pelo SAS para modelagem de dados. As etapas incluem:

1. **Sample:** Amostragem dos dados.
2. **Explore:** Exploração dos dados para encontrar padrões.
3. **Modify:** Modificação e transformação dos dados.
4. **Model:** Construção dos modelos preditivos.
5. **Assess:** Avaliação dos modelos.
![SEMMA](https://documentation.sas.com/api/docsets/emref/14.3/content/images/semma.png?locale=en)

## Estrutura do Projeto

O projeto está organizado nas seguintes pastas principais:

- **data:** Contém o banco de dados utilizado no projeto.
- **etl:** Processos de extração, transformação e carregamento de dados.
  - Contém queries SQL utilizadas para a criação de tabelas e a preparação dos dados.
- **model_sub:** Contém scripts e modelos para a predição.
  - **predict:** Scripts e queries SQL utilizados na extração dos dados.
  - **ml:** Scripts de predição e avaliação dos modelos.
- **train:** Scripts e queries SQL para o treinamento dos modelos.
  - Inclui a preparação dos dados para treinamento e os scripts de modelagem.
- **models:** Modelos preditivos treinados salvos em formato pickle.


