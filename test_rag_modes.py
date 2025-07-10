#!/usr/bin/env python3
"""
Script de Test et Comparaison des Modes RAG

Ce script permet de tester et comparer les différentes stratégies RAG
pour évaluer leurs performances relatives.

Usage:
    python test_rag_modes.py                    # Test complet automatique
    python test_rag_modes.py --interactive      # Mode interactif
    python test_rag_modes.py --strategy naive   # Test d'une stratégie spécifique
    python test_rag_modes.py --benchmark        # Benchmark détaillé
"""

import argparse
import json
import time
from typing import List, Dict, Any
from pathlib import Path

from enhanced_rag_app import EnhancedRAGApplication, RAGComparison


class RAGTester:
    """Classe pour tester et comparer les stratégies RAG."""
    
    def __init__(self):
        """Initialise le testeur RAG."""
        self.app = EnhancedRAGApplication()
        self.test_results = []
        self.benchmark_questions = [
            {
                "question": "What is an AI agent and how does it work?",
                "category": "Definition",
                "expected_topics": ["agent", "AI", "components", "workflow"]
            },
            {
                "question": "How does memory work in language model agents?",
                "category": "Technical",
                "expected_topics": ["memory", "storage", "retrieval", "context"]
            },
            {
                "question": "What are the main challenges in agent planning?",
                "category": "Analysis",
                "expected_topics": ["planning", "challenges", "decomposition", "execution"]
            },
            {
                "question": "Can you explain the difference between reactive and deliberative agents?",
                "category": "Comparison",
                "expected_topics": ["reactive", "deliberative", "differences", "behavior"]
            },
            {
                "question": "What tools and frameworks are mentioned for building AI agents?",
                "category": "Practical",
                "expected_topics": ["tools", "frameworks", "implementation", "development"]
            }
        ]
    
    def setup_application(self):
        """Configure l'application RAG pour les tests."""
        print("🔧 Configuration de l'application de test...")
        
        # Initialisation complète
        self.app.setup_environment()
        self.app.initialize_models()
        self.app.setup_vector_store()
        self.app.initialize_strategies()
        
        # Chargement des documents de test
        print("📥 Chargement des documents de test...")
        test_sources = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/"
        ]
        
        for source in test_sources:
            try:
                documents = self.app.load_and_process_documents(source)
                self.app.populate_vector_store(documents)
                print(f"✅ Document chargé: {source}")
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {source}: {e}")
        
        print("✅ Application configurée pour les tests")
    
    def test_single_strategy(self, strategy_name: str, questions: List[str] = None) -> Dict[str, Any]:
        """
        Teste une stratégie spécifique.
        
        Args:
            strategy_name: Nom de la stratégie à tester
            questions: Questions à tester (utilise benchmark si None)
            
        Returns:
            Résultats du test
        """
        if questions is None:
            questions = [q["question"] for q in self.benchmark_questions]
        
        print(f"\n🎯 Test de la stratégie: {strategy_name}")
        print("=" * 60)
        
        if strategy_name not in self.app.list_strategies():
            print(f"❌ Stratégie '{strategy_name}' non disponible")
            return {}
        
        # Configuration de la stratégie
        self.app.set_strategy(strategy_name)
        
        results = {
            "strategy": strategy_name,
            "questions_tested": len(questions),
            "results": [],
            "summary": {}
        }
        
        total_time = 0
        successful_queries = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}/{len(questions)}: {question}")
            
            try:
                start_time = time.time()
                result = self.app.ask_question(question)
                end_time = time.time()
                
                query_result = {
                    "question": question,
                    "response_length": len(result.response),
                    "sources_count": len(result.sources),
                    "processing_time": result.processing_time,
                    "success": True
                }
                
                results["results"].append(query_result)
                total_time += result.processing_time
                successful_queries += 1
                
                print(f"✅ Traité en {result.processing_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Erreur: {e}")
                query_result = {
                    "question": question,
                    "error": str(e),
                    "success": False
                }
                results["results"].append(query_result)
        
        # Calcul du résumé
        if successful_queries > 0:
            results["summary"] = {
                "success_rate": successful_queries / len(questions),
                "avg_processing_time": total_time / successful_queries,
                "total_processing_time": total_time,
                "successful_queries": successful_queries
            }
        
        self._display_strategy_results(results)
        return results
    
    def compare_all_strategies(self, questions: List[str] = None) -> List[RAGComparison]:
        """
        Compare toutes les stratégies disponibles.
        
        Args:
            questions: Questions à tester (utilise benchmark si None)
            
        Returns:
            Liste des comparaisons
        """
        if questions is None:
            questions = [q["question"] for q in self.benchmark_questions]
        
        print(f"\n🆚 COMPARAISON DE TOUTES LES STRATÉGIES")
        print("=" * 80)
        print(f"Questions à tester: {len(questions)}")
        print(f"Stratégies disponibles: {', '.join(self.app.list_strategies())}")
        
        comparisons = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"QUESTION {i}/{len(questions)}: {question}")
            print('='*60)
            
            try:
                comparison = self.app.compare_strategies(question)
                comparisons.append(comparison)
                
                # Pause pour lecture si mode interactif
                if len(questions) <= 3:
                    input("\nAppuyez sur Entrée pour continuer...")
                
            except Exception as e:
                print(f"❌ Erreur lors de la comparaison: {e}")
                continue
        
        # Analyse globale
        self._analyze_global_results(comparisons)
        
        return comparisons
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """
        Effectue un benchmark détaillé de toutes les stratégies.
        
        Returns:
            Résultats du benchmark
        """
        print(f"\n⚡ BENCHMARK DE PERFORMANCE")
        print("=" * 80)
        
        benchmark_results = {
            "strategies": {},
            "questions": [],
            "global_analysis": {}
        }
        
        # Test de chaque stratégie
        for strategy in self.app.list_strategies():
            print(f"\n🎯 Benchmark de {strategy}...")
            
            strategy_results = self.test_single_strategy(
                strategy, 
                [q["question"] for q in self.benchmark_questions]
            )
            
            benchmark_results["strategies"][strategy] = strategy_results
        
        # Questions testées
        benchmark_results["questions"] = self.benchmark_questions
        
        # Analyse globale
        benchmark_results["global_analysis"] = self._calculate_global_metrics(
            benchmark_results["strategies"]
        )
        
        # Sauvegarde des résultats
        self._save_benchmark_results(benchmark_results)
        
        # Affichage du résumé final
        self._display_benchmark_summary(benchmark_results)
        
        return benchmark_results
    
    def interactive_mode(self):
        """Mode interactif pour tester les stratégies."""
        print(f"\n🎮 MODE INTERACTIF")
        print("=" * 60)
        
        while True:
            print(f"\nOptions disponibles:")
            print("1. Tester une stratégie spécifique")
            print("2. Comparer toutes les stratégies sur une question")
            print("3. Changer de stratégie active")
            print("4. Poser une question personnalisée")
            print("5. Afficher le résumé des performances")
            print("6. Quitter")
            
            choice = input("\nVotre choix (1-6): ").strip()
            
            if choice == "1":
                self._interactive_test_strategy()
            elif choice == "2":
                self._interactive_compare_strategies()
            elif choice == "3":
                self._interactive_change_strategy()
            elif choice == "4":
                self._interactive_custom_question()
            elif choice == "5":
                self.app.display_performance_summary()
            elif choice == "6":
                print("👋 Au revoir!")
                break
            else:
                print("❌ Choix invalide")
    
    def _interactive_test_strategy(self):
        """Test interactif d'une stratégie."""
        strategies = self.app.list_strategies()
        print(f"\nStratégies disponibles: {', '.join(strategies)}")
        
        strategy = input("Quelle stratégie tester? ").strip()
        if strategy not in strategies:
            print(f"❌ Stratégie '{strategy}' non disponible")
            return
        
        print("Questions benchmark ou personnalisée?")
        print("1. Questions benchmark")
        print("2. Question personnalisée")
        
        choice = input("Votre choix (1-2): ").strip()
        
        if choice == "1":
            self.test_single_strategy(strategy)
        elif choice == "2":
            question = input("Votre question: ").strip()
            if question:
                self.test_single_strategy(strategy, [question])
    
    def _interactive_compare_strategies(self):
        """Comparaison interactive des stratégies."""
        question = input("Question pour la comparaison: ").strip()
        if question:
            try:
                self.app.compare_strategies(question)
            except Exception as e:
                print(f"❌ Erreur: {e}")
    
    def _interactive_change_strategy(self):
        """Changement interactif de stratégie."""
        strategies = self.app.list_strategies()
        current = self.app.get_current_strategy()
        
        print(f"\nStratégie actuelle: {current}")
        print(f"Disponibles: {', '.join(strategies)}")
        
        strategy = input("Nouvelle stratégie: ").strip()
        if strategy in strategies:
            try:
                self.app.set_strategy(strategy)
                print(f"✅ Stratégie changée vers: {strategy}")
            except Exception as e:
                print(f"❌ Erreur: {e}")
        else:
            print(f"❌ Stratégie '{strategy}' non disponible")
    
    def _interactive_custom_question(self):
        """Question personnalisée interactive."""
        question = input("Votre question: ").strip()
        if question:
            try:
                result = self.app.ask_question(question)
                print(f"\n📊 Métadonnées:")
                print(f"   Temps de traitement: {result.processing_time:.2f}s")
                print(f"   Nombre de sources: {len(result.sources)}")
                print(f"   Stratégie utilisée: {result.strategy_used}")
            except Exception as e:
                print(f"❌ Erreur: {e}")
    
    def _display_strategy_results(self, results: Dict[str, Any]):
        """Affiche les résultats d'une stratégie."""
        print(f"\n📊 RÉSULTATS - {results['strategy']}")
        print("=" * 50)
        
        if "summary" in results and results["summary"]:
            summary = results["summary"]
            print(f"Taux de succès: {summary['success_rate']:.1%}")
            print(f"Temps moyen: {summary['avg_processing_time']:.2f}s")
            print(f"Temps total: {summary['total_processing_time']:.2f}s")
            print(f"Requêtes réussies: {summary['successful_queries']}/{results['questions_tested']}")
        else:
            print("Aucun résultat valide")
    
    def _analyze_global_results(self, comparisons: List[RAGComparison]):
        """Analyse globale des résultats de comparaison."""
        if not comparisons:
            print("Aucune comparaison à analyser")
            return
        
        print(f"\n📈 ANALYSE GLOBALE")
        print("=" * 50)
        
        # Comptage des victoires par stratégie
        strategy_wins = {}
        strategy_times = {}
        
        for comparison in comparisons:
            best = comparison.best_strategy
            if best in strategy_wins:
                strategy_wins[best] += 1
            else:
                strategy_wins[best] = 1
            
            # Temps de traitement
            for strategy, result in comparison.results.items():
                if strategy not in strategy_times:
                    strategy_times[strategy] = []
                strategy_times[strategy].append(result.processing_time)
        
        # Affichage des statistiques
        print("🏆 Victoires par stratégie:")
        for strategy, wins in sorted(strategy_wins.items(), key=lambda x: x[1], reverse=True):
            print(f"   {strategy}: {wins}/{len(comparisons)} ({wins/len(comparisons):.1%})")
        
        print("\n⚡ Temps de traitement moyens:")
        for strategy, times in strategy_times.items():
            avg_time = sum(times) / len(times)
            print(f"   {strategy}: {avg_time:.2f}s")
    
    def _calculate_global_metrics(self, strategies_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les métriques globales du benchmark."""
        global_metrics = {
            "best_overall_strategy": None,
            "fastest_strategy": None,
            "most_reliable_strategy": None,
            "detailed_comparison": {}
        }
        
        strategy_scores = {}
        
        for strategy_name, results in strategies_results.items():
            if "summary" not in results or not results["summary"]:
                continue
            
            summary = results["summary"]
            
            # Score composite
            score = (
                summary["success_rate"] * 0.4 +  # 40% fiabilité
                (1 / summary["avg_processing_time"]) * 0.3 +  # 30% rapidité
                summary["successful_queries"] * 0.3  # 30% volume
            )
            
            strategy_scores[strategy_name] = {
                "composite_score": score,
                "success_rate": summary["success_rate"],
                "avg_time": summary["avg_processing_time"],
                "total_success": summary["successful_queries"]
            }
        
        if strategy_scores:
            # Meilleure stratégie globale
            global_metrics["best_overall_strategy"] = max(
                strategy_scores.keys(), 
                key=lambda x: strategy_scores[x]["composite_score"]
            )
            
            # Stratégie la plus rapide
            global_metrics["fastest_strategy"] = min(
                strategy_scores.keys(),
                key=lambda x: strategy_scores[x]["avg_time"]
            )
            
            # Stratégie la plus fiable
            global_metrics["most_reliable_strategy"] = max(
                strategy_scores.keys(),
                key=lambda x: strategy_scores[x]["success_rate"]
            )
            
            global_metrics["detailed_comparison"] = strategy_scores
        
        return global_metrics
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Sauvegarde les résultats du benchmark."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Résultats sauvegardés dans: {filename}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def _display_benchmark_summary(self, results: Dict[str, Any]):
        """Affiche le résumé final du benchmark."""
        print(f"\n🏁 RÉSUMÉ FINAL DU BENCHMARK")
        print("=" * 80)
        
        global_analysis = results["global_analysis"]
        
        if global_analysis:
            print(f"🥇 Meilleure stratégie globale: {global_analysis['best_overall_strategy']}")
            print(f"⚡ Stratégie la plus rapide: {global_analysis['fastest_strategy']}")
            print(f"🛡️  Stratégie la plus fiable: {global_analysis['most_reliable_strategy']}")
            
            print(f"\n📊 Scores détaillés:")
            for strategy, metrics in global_analysis["detailed_comparison"].items():
                print(f"   {strategy}:")
                print(f"      Score composite: {metrics['composite_score']:.3f}")
                print(f"      Taux de succès: {metrics['success_rate']:.1%}")
                print(f"      Temps moyen: {metrics['avg_time']:.2f}s")
        else:
            print("Aucune donnée d'analyse disponible")


def main():
    """Point d'entrée principal du script de test."""
    parser = argparse.ArgumentParser(description="Test et comparaison des modes RAG")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    parser.add_argument("--strategy", type=str, help="Tester une stratégie spécifique")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark complet")
    parser.add_argument("--question", type=str, help="Question personnalisée")
    
    args = parser.parse_args()
    
    # Initialisation du testeur
    tester = RAGTester()
    
    try:
        # Configuration
        tester.setup_application()
        
        if args.interactive:
            # Mode interactif
            tester.interactive_mode()
            
        elif args.strategy:
            # Test d'une stratégie spécifique
            questions = [args.question] if args.question else None
            tester.test_single_strategy(args.strategy, questions)
            
        elif args.benchmark:
            # Benchmark complet
            tester.benchmark_performance()
            
        elif args.question:
            # Comparaison sur une question spécifique
            tester.compare_all_strategies([args.question])
            
        else:
            # Test automatique complet
            print("🚀 Lancement du test automatique complet...")
            tester.compare_all_strategies()
            
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()