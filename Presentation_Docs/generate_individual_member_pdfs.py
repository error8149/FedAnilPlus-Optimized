from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

def generate_member_pdf(member_name, role, tasks, speaking_points, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=22, alignment=1, spaceAfter=10
    )
    header_style = ParagraphStyle(
        'HeaderStyle', parent=styles['Heading2'], fontSize=16, color=colors.HexColor("#2E5984"), spaceBefore=10, spaceAfter=8
    )

    content = []
    content.append(Paragraph(f"Presentation Guide: {member_name}", title_style))
    content.append(Paragraph(f"Role: {role}", styles['Normal']))
    content.append(Spacer(1, 0.3 * inch))

    # Detailed Contributions
    content.append(Paragraph("Technical Contributions", header_style))
    for t in tasks:
        content.append(Paragraph(f"• {t}", styles['Normal']))
        content.append(Spacer(1, 0.15 * inch))
    
    content.append(Spacer(1, 0.3 * inch))

    # Speaking Guide
    content.append(Paragraph("Speaking Points & Presentation Flow", header_style))
    for p in speaking_points:
        content.append(Paragraph(f"➜ {p}", styles['Normal']))
        content.append(Spacer(1, 0.2 * inch))


    content.append(Spacer(1, 0.5 * inch))
    content.append(Paragraph("Tip: Use the 'show_team_work.py' conductor script during the live demo to keep track of timing.", styles['Italic']))

    doc.build(content)
    print(f"Generated: {filename}")

def main():
    team_data = [
        {
            "name": "Muhammad Hadi",
            "role": "System & Performance Engineer",
            "tasks": [
                "GPU Acceleration: Implementation of CUDA detection and CuDNN benchmarking for 10x-15x speedup.",
                "Mixed Precision (AMP): Integration of torch.cuda.amp (GradScaler/Autocast) reducing memory and doubling speed.",
                "Optimizer Tuning: Implemented zero_grad(set_to_none=True) for efficient VRAM utilization."
            ],
            "points": [
                "<b>Script:</b> 'Hello. My role was to solve the bottleneck of processing speed. Initially, the simulation took hours. I configured the system to use the NVIDIA MX550 GPU instead of the CPU.'",
                "<b>Script:</b> 'I also implemented Mixed Precision Training. This allows the GPU to use half-precision math (FP16), which is much faster while keeping the same accuracy.'",
                "<b>What to Show:</b> Point to the terminal output where it says <i>Using device: cuda:0</i> and the number of GPUs available.'",
                "<b>Conclusion:</b> 'The result is a 10x to 15x increase in speed, making long simulations possible in minutes.'"
            ],
            "file": "Member_Report_Hadi.pdf"
        },
        {
            "name": "Muhammad Jahangir",
            "role": "AI & Optimization Specialist",
            "tasks": [
                "Batch Normalization: Strategically added BatchNorm2d layers to improve convergence and boost accuracy by +5%.",
                "Weight Decay (L2): Implemented regularization to prevent overfitting on non-IID datasets.",
                "Gradient Clipping: Added global norm clipping to stabilize training against malicious enterprise updates."
            ],
            "points": [
                "<b>Script:</b> 'I focused on the quality of the learning process. To make the model more accurate, I added Batch Normalization layers. This stabilizes the internal learning of the neural network.'",
                "<b>Script:</b> 'I also added Weight Decay, which is a mathematical penalty that prevents the model from overfitting or essentially 'memorizing' the training data.'",
                "<b>What to Show:</b> Show the code in <i>Models.py</i> where <i>BatchNorm2d</i> is added after convolutions.'",
                "<b>Conclusion:</b> 'Together, these changes boosted our final accuracy and made the training curve much smoother.'"
            ],
            "file": "Member_Report_Jahangir.pdf"
        },
        {
            "name": "Talha Safique",
            "role": "Reliability & Quality Engineer",
            "tasks": [
                "OOM (Out of Memory) Fix: Created a hybrid deep-copy system (_deepcopy_to_cpu) for low-VRAM GPUs.",
                "Clustering Bug Fix: Resolved critical CUDA-to-NumPy conversion error in the KMedoids algorithm.",
                "Lifecycle Management: Implemented periodic VRAM cache clearing and system-wide documentation."
            ],
            "points": [
                "<b>Script:</b> 'My focus was system reliability. Running 100 enterprise nodes on a 2GB VRAM card lead to 'Out of Memory' crashes. I solved this by creating a hybrid memory manager.'",
                "<b>Script:</b> 'My custom <i>_deepcopy_to_cpu</i> function moves model data to system RAM when copying, keeping the precious GPU memory free for active training.'",
                "<b>What to Show:</b> Open the <i>CHANGES_DOCUMENTATION.md</i> and show the <i>_deepcopy_to_cpu</i> helper function.'",
                "<b>Conclusion:</b> 'I also fixed a critical scikit-learn bug where GPU tensors were crashing the KMedoids algorithm. Now the system is 100% stable.'"
            ],
            "file": "Member_Report_Talha.pdf"
        }
    ]


    for member in team_data:
        generate_member_pdf(member['name'], member['role'], member['tasks'], member['points'], member['file'])

if __name__ == "__main__":
    main()
