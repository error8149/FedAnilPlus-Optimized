from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_pdf():
    filename = "FedAnilPlus_Project_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=1, # Center
        spaceAfter=20
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontSize=18,
        color=colors.HexColor("#2E5984"),
        spaceBefore=15,
        spaceAfter=10
    )

    content = []

    # Title
    content.append(Paragraph("FedAnilPlus: System Optimization Report", title_style))
    content.append(Paragraph("Team Work Distribution & Technical Improvements", styles['Normal']))
    content.append(Spacer(1, 0.5 * inch))

    # Highlights Section
    content.append(Paragraph("Key Highlights", header_style))
    highlights = [
        "• Stability: Running 100 enterprise nodes on 2GB VRAM (MX550).",
        "• Accuracy: Normalized convergence via Batch Normalization (+5%).",
        "• Efficiency: 10x-15x speed improvement using CUDA & Mixed Precision."
    ]
    for h in highlights:
        content.append(Paragraph(h, styles['Normal']))
    content.append(Spacer(1, 0.3 * inch))

    # Detailed Task Data
    hadi_tasks = [
        "<b>GPU Acceleration & System Speedup:</b> Configured PyTorch to utilize the NVIDIA GeForce MX550 GPU. Implemented <i>cudnn.benchmark=True</i>, resulting in a 10x-15x speed increase.",
        "<b>Mixed Precision Training (AMP):</b> Integrated <i>torch.cuda.amp</i> (GradScaler and autocast) for FP16 training, reducing memory footprint and boosting efficiency by 2x.",
        "<b>Optimizer Memory Tuning:</b> Optimized gradient resets using <i>set_to_none=True</i>, minimizing memory write operations during the training loop."
    ]
    
    jahangir_tasks = [
        "<b>Neural Network Architecture Improvement:</b> Integrated <i>BatchNorm2d</i> layers, reducing internal covariate shift and boosting model accuracy by +5%.",
        "<b>Optimization Regularization:</b> Implemented L2 weight decay to penalize large weights, significantly reducing overfitting on the FEMINIST and CIFAR-10 datasets.",
        "<b>Training Stability:</b> Integrated global <i>Gradient Clipping</i> to prevent gradient explosion, ensuring stable convergence in the asymmetric federated setting."
    ]
    
    talha_tasks = [
        "<b>GPU Memory Management & OOM Fix:</b> Engineered a hybrid CPU/GPU deep-copy system (<i>_deepcopy_to_cpu</i>) to allow 100-node simulation on a 2GB VRAM card.",
        "<b>Multi-Environment Compatibility:</b> Resolved scikit-learn/CUDA tensor bugs by implementing automated CPU fallback for clustering operations.",
        "<b>System Reliability & Lifecycle:</b> Implemented periodic cache clearing and comprehensive performance documentation for future scalability."
    ]

    # Team Work Distribution Table
    content.append(Paragraph("Project Work Distribution", header_style))
    
    # Task formatting for table
    def format_tasks(task_list):
        return "<br/>".join([f"&bull; {t}" for t in task_list])

    # Custom style for table cells with tighter leading and bullet support
    table_cell_style = ParagraphStyle(
        'TableCellStyle',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        spaceBefore=0,
        spaceAfter=0,
        alignment=0 # Left
    )


    team_data = [
        ["Member", "Detailed Contributions"],
        [Paragraph("<b>Muhammad Hadi</b><br/>(System & Performance)", table_cell_style), 
         Paragraph(format_tasks(hadi_tasks), table_cell_style)],
        [Paragraph("<b>Muhammad Jahangir</b><br/>(AI & Optimization)", table_cell_style), 
         Paragraph(format_tasks(jahangir_tasks), table_cell_style)],
        [Paragraph("<b>Talha Safique</b><br/>(Reliability & Fixes)", table_cell_style), 
         Paragraph(format_tasks(talha_tasks), table_cell_style)]
    ]
    
    # Adjusted widths: 1.4 for name, 5.1 for tasks (Total 6.5")
    table = Table(team_data, colWidths=[1.4*inch, 5.1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2E5984")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    content.append(table)


    content.append(Spacer(1, 0.4 * inch))

    # Summary
    content.append(Paragraph("Final Performance Benchmarks", header_style))
    content.append(Paragraph("<b>Stability:</b> 100% success rate on 2GB MX550 over 50 rounds.", styles['Normal']))
    content.append(Paragraph("<b>Efficiency:</b> Communication rounds reduced from hours to minutes via mixed precision & GPU acceleration.", styles['Normal']))

    doc.build(content)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    generate_pdf()
