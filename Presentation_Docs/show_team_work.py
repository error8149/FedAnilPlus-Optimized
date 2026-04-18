import time

def print_header(text):
    print("\n" + "="*60)
    print(f" {text.center(58)} ")
    print("="*60)

def show_guide():
    print_header("FedAnilPlus Presentation Conductor")
    print("This script will guide you through your presentation.")
    print("Press ENTER to move to the next member's part.\n")

    presentation = [
        {
            "member": "Muhammad Hadi",
            "role": "System & Performance Engineer",
            "points": [
                "Focus: Hardware acceleration and raw speed.",
                "Mention: how you moved training from CPU to NVIDIA GPU.",
                "Mention: Mixed Precision (AMP) and why it's 2x faster.",
                "Show: 'python main.py' output where it detects 'cuda:0'."
            ],
            "show_when": "Start of the presentation after Intro."
        },
        {
            "member": "Muhammad Jahangir",
            "role": "AI & Optimization Specialist",
            "points": [
                "Focus: Improving accuracy and training stability.",
                "Mention: Why Batch Normalization was added to the CNN.",
                "Mention: How Weight Decay reduced overfitting.",
                "Show: 'accuracy_comm_*.txt' logs showing increasing percentage."
            ],
            "show_when": "Middle part, after the system is proved to be fast."
        },
        {
            "member": "Talha Safique",
            "role": "Reliability & Quality Engineer",
            "points": [
                "Focus: Memory management and fixing critical bugs.",
                "Mention: The hybrid CPU/GPU deep-copy fix (the OOM solution).",
                "Mention: Fixing the KMedoids crash for the whole team.",
                "Show: 'CHANGES_DOCUMENTATION.md' or the detailed project PDF."
            ],
            "show_when": "Conclusion - proving the system is stable and bug-free."
        }
    ]

    for part in presentation:
        print(f"\n>>> NEXT UP: {part['member']} ({part['role']})")
        print(f"    When to speak: {part['show_when']}")
        print("-" * 30)
        for i, point in enumerate(part['points'], 1):
            print(f"    {i}. {point}")
        
        input("\n[Press ENTER for next speaker...]")

    print_header("Final Wrap-up: Show the Final Benchmarks PDF!")
    print("Presentation Finished. Good luck!")

if __name__ == "__main__":
    show_guide()

